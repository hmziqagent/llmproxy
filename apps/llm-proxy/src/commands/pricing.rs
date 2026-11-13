//! `llm-proxy pricing sync`: the opt-in models.dev pricing sync (plan 05
//! step 9, backlog 22).
//!
//! A provider opts in with `[provider.pricing_sync] models_dev_id = "..."`
//! naming its id in the models.dev catalogue. The command fetches
//! `https://models.dev/api.json` (schema verified live on 2026-11-06: a map
//! of provider id → `{ models: { <model id>: { cost: { input, output,
//! cache_read, cache_write, ... } } } }` with costs in USD per **million**
//! tokens), converts every priced model to exact per-token decimals, and
//! rewrites the provider file's `[provider.pricing]` table in place with
//! [`toml_edit`], leaving the rest of the file — comments included —
//! untouched.
//!
//! Models the catalogue does not price are left out: unpriced is a state
//! (an absent entry), never a silent `$0` — the same rule the usage
//! summaries enforce (BUG-24). A sync replaces the whole pricing table, so
//! manual entries do not survive it; operators who hand-tune prices do not
//! opt the provider in.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use llm_proxy_core::{ModelPricing, ProviderFile};
use llm_proxy_transport::DiscoveryClient;
use rust_decimal::Decimal;
use serde_json::Value;
use toml_edit::{DocumentMut, Item, Table, value};

use crate::config_validation::validate_toml_extension;
use crate::paths::{providers_dir_for, resolve_config};

/// The default catalogue URL.
const DEFAULT_MODELS_DEV_URL: &str = "https://models.dev/api.json";

/// Ceiling on the catalogue download (the real file is a few MB).
const MAX_CATALOGUE_BYTES: usize = 32 * 1024 * 1024;

/// Run `pricing sync`.
///
/// # Errors
///
/// `anyhow` when the config is unusable, no opted-in provider matches, the
/// catalogue cannot be fetched or parsed, or a provider file cannot be
/// rewritten.
pub async fn cmd_pricing_sync(
    config_path: Option<PathBuf>,
    provider_filter: Option<String>,
    url: Option<String>,
    dry_run: bool,
) -> Result<()> {
    let path = resolve_config(config_path.as_deref());
    validate_toml_extension(&path)?;
    let providers_dir = providers_dir_for(&path);
    if !providers_dir.exists() {
        bail!(
            "providers directory {} does not exist",
            providers_dir.display()
        );
    }

    // Every provider file that opted in (and matches the filter, if any).
    let mut targets: Vec<(PathBuf, String)> = Vec::new();
    for file in provider_files(&providers_dir)? {
        let raw = std::fs::read_to_string(&file)?;
        let parsed: ProviderFile =
            toml::from_str(&raw).with_context(|| format!("parsing {}", file.display()))?;
        let Some(sync) = &parsed.provider.pricing_sync else {
            continue;
        };
        if let Some(filter) = &provider_filter
            && &parsed.provider.name != filter
        {
            continue;
        }
        targets.push((file, sync.models_dev_id.clone()));
    }
    if targets.is_empty() {
        println!(
            "No provider with [provider.pricing_sync] found in {}.",
            providers_dir.display()
        );
        println!("Opt a provider in with:");
        println!();
        println!("  [provider.pricing_sync]");
        println!("  models_dev_id = \"anthropic\"");
        return Ok(());
    }

    let url = url.unwrap_or_else(|| DEFAULT_MODELS_DEV_URL.to_owned());
    let catalogue = fetch_catalogue(&url).await?;

    for (file, models_dev_id) in &targets {
        let (entries, skipped) = pricing_entries(&catalogue, models_dev_id)?;
        if entries.is_empty() {
            println!(
                "  {}: no priced models for models.dev id \"{}\"; file left unchanged",
                file.display(),
                models_dev_id
            );
            print_skipped_partial(&skipped);
            continue;
        }
        let preview = describe(&entries);
        if dry_run {
            println!(
                "  {} (dry run): would write {} priced models{}",
                file.display(),
                entries.len(),
                preview
            );
            print_skipped_partial(&skipped);
            continue;
        }
        let source =
            std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        let updated = apply_pricing(&source, &entries, &url)?;
        std::fs::write(file, updated).with_context(|| format!("writing {}", file.display()))?;
        println!(
            "  {}: wrote {} priced models{}",
            file.display(),
            entries.len(),
            preview
        );
        print_skipped_partial(&skipped);
    }
    Ok(())
}

/// Reports the models a sync left unpriced because the catalogue prices only
/// one of their input/output dimensions, so the omission is visible instead
/// of a silent `$0` dimension.
fn print_skipped_partial(skipped: &[String]) {
    if skipped.is_empty() {
        return;
    }
    println!(
        "    skipped {} model(s) with a partial price (unpriced): {}",
        skipped.len(),
        skipped.join(", ")
    );
}

fn provider_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension == "toml")
        })
        .collect();
    files.sort();
    Ok(files)
}

async fn fetch_catalogue(url: &str) -> Result<Value> {
    let client =
        DiscoveryClient::try_new().context("building the HTTP client for the catalogue fetch")?;
    client
        .fetch_json(url, MAX_CATALOGUE_BYTES)
        .await
        .with_context(|| format!("fetching the pricing catalogue from {url}"))
}

/// The priced models one models.dev provider declares, as per-token USD
/// decimals keyed by upstream model id, plus the model ids skipped because
/// only one of `cost.input` and `cost.output` is priced: a partial price
/// would silently charge `$0` for the missing dimension, so the model stays
/// unpriced and is reported instead. Models without any priced dimension at
/// all are also left out (unpriced, never `$0`), but are not partial prices.
///
/// # Errors
///
/// `anyhow` when the catalogue entry for `models_dev_id` is malformed.
fn pricing_entries(
    catalogue: &Value,
    models_dev_id: &str,
) -> Result<(Vec<(String, ModelPricing)>, Vec<String>)> {
    let Some(provider) = catalogue.get(models_dev_id) else {
        bail!("the catalogue has no provider \"{models_dev_id}\"");
    };
    let Some(models) = provider.get("models").and_then(Value::as_object) else {
        bail!(
            "catalogue entry for \"{models_dev_id}\" has no models table; is this models.dev api.json?"
        );
    };
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    for (model_id, model) in models {
        let Some(cost) = model.get("cost") else {
            continue;
        };
        let input = per_token(cost.get("input"))?;
        let output = per_token(cost.get("output"))?;
        match (input, output) {
            (Some(input), Some(output)) => entries.push((
                model_id.clone(),
                ModelPricing {
                    input,
                    output,
                    // Cache dimensions absent from the catalogue stay 0: a
                    // model that reports no cache pricing reports no cache
                    // tokens.
                    cache_creation: per_token(cost.get("cache_write"))?.unwrap_or(Decimal::ZERO),
                    cache_read: per_token(cost.get("cache_read"))?.unwrap_or(Decimal::ZERO),
                    // models.dev prices reasoning as output, so there is no
                    // separate reasoning dimension to carry.
                    reasoning: Decimal::ZERO,
                },
            )),
            (Some(_), None) | (None, Some(_)) => {
                // Exactly one dimension priced: writing the other as a
                // silent $0 would misprice every request, so the model is
                // unpriced and reported as skipped.
                skipped.push(model_id.clone());
            }
            (None, None) => {} // no priced dimensions at all: unpriced, not free
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    skipped.sort();
    Ok((entries, skipped))
}

/// One catalogue price (USD per million tokens) as an exact per-token
/// decimal, or `None` when the field is absent. Numbers are parsed through
/// their shortest round-trip text so no float ever touches the money; that
/// text can be exponent form (`1e-6` is serde_json's rendering of
/// `0.000001`), which the plain-decimal parser rejects.
fn per_token(field: Option<&Value>) -> Result<Option<Decimal>> {
    let Some(field) = field else {
        return Ok(None);
    };
    let text = match field {
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        other => bail!("catalogue price is neither a number nor a string: {other}"),
    };
    let per_million = Decimal::from_str_exact(&text)
        .or_else(|_| Decimal::from_scientific(&text))
        .with_context(|| format!("catalogue price {text:?} is not an exact decimal"))?;
    let per_token = per_million
        .checked_div(Decimal::from(1_000_000u64))
        .context("dividing the catalogue price by one million")?;
    Ok(Some(per_token))
}

/// Replaces `[provider.pricing]` in the provider file with the synced
/// entries, preserving everything else in the document.
///
/// # Errors
///
/// `anyhow` when the file is not valid TOML or has no `[provider]` table.
fn apply_pricing(source: &str, entries: &[(String, ModelPricing)], url: &str) -> Result<String> {
    let mut document: DocumentMut = source
        .parse()
        .with_context(|| "parsing the provider file as TOML before the pricing rewrite")?;
    let provider = document
        .get_mut("provider")
        .and_then(Item::as_table_like_mut)
        .context("the provider file has no [provider] table")?;

    let mut pricing = Table::new();
    pricing.set_implicit(true);
    let now = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_owned());
    pricing.decor_mut().set_prefix(format!(
        "\n# Synced from {url} at {now}. This sync replaces the whole\n\
         # [provider.pricing] table; hand-tuned entries do not survive it.\n"
    ));
    for (model, prices) in entries {
        let mut table = Table::new();
        table.decor_mut().set_prefix("\n");
        table.insert("input", value(prices.input.to_string()));
        table.insert("output", value(prices.output.to_string()));
        if prices.cache_creation != Decimal::ZERO {
            table.insert("cache_creation", value(prices.cache_creation.to_string()));
        }
        if prices.cache_read != Decimal::ZERO {
            table.insert("cache_read", value(prices.cache_read.to_string()));
        }
        pricing.insert(model.as_str(), Item::Table(table));
    }
    provider.insert("pricing", Item::Table(pricing));
    Ok(document.to_string())
}

fn describe(entries: &[(String, ModelPricing)]) -> String {
    let mut sample = entries
        .first()
        .map(|(model, _)| format!(" (e.g. {model})"))
        .unwrap_or_default();
    if entries.len() > 1 {
        let last = entries
            .last()
            .map(|(model, _)| model.as_str())
            .unwrap_or_default();
        sample.push_str(&format!(" through {last}"));
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use llm_proxy_core::ModelId;

    fn entries_as_map(entries: &[(String, ModelPricing)]) -> HashMap<ModelId, ModelPricing> {
        entries
            .iter()
            .map(|(model, pricing)| (ModelId::new(model.clone()), pricing.clone()))
            .collect()
    }

    fn sample_catalogue() -> Value {
        let raw = include_str!("../../tests/fixtures/models-dev-sample.json");
        serde_json::from_str(raw).expect("fixture parses")
    }

    #[test]
    fn per_million_prices_become_exact_per_token_decimals() {
        let (entries, _) = pricing_entries(&sample_catalogue(), "anthropic").expect("entries");
        let map = entries_as_map(&entries);
        let claude = map
            .get(&ModelId::new("claude-sonnet-4-5"))
            .expect("priced model present");
        // 3 USD / 1M input, 15 USD / 1M output, 0.3 cache_read,
        // 3.75 cache_write (models.dev spells creation "cache_write").
        assert_eq!(claude.input.normalize().to_string(), "0.000003");
        assert_eq!(claude.output.normalize().to_string(), "0.000015");
        assert_eq!(claude.cache_read.normalize().to_string(), "0.0000003");
        assert_eq!(claude.cache_creation.normalize().to_string(), "0.00000375");
        assert_eq!(claude.reasoning, Decimal::ZERO);
    }

    #[test]
    fn unpriced_models_stay_unpriced_never_zero() {
        let (entries, skipped) =
            pricing_entries(&sample_catalogue(), "anthropic").expect("entries");
        let map = entries_as_map(&entries);
        assert!(
            !map.contains_key(&ModelId::new("unpriced-preview")),
            "a model with no cost object must not gain a $0 entry"
        );
        assert!(
            !map.contains_key(&ModelId::new("no-cost-dimensions")),
            "a model whose cost has no priced dimensions stays unpriced"
        );
        assert!(
            !skipped.contains(&"unpriced-preview".to_owned()),
            "a model with no cost object has no partial price to report"
        );
        assert!(
            !skipped.contains(&"no-cost-dimensions".to_owned()),
            "a model with no priced dimensions is unpriced, not partially priced"
        );
        assert!(
            map.values()
                .all(|pricing| pricing.input != Decimal::ZERO && pricing.output != Decimal::ZERO),
            "no synced entry may carry a silent $0 dimension: {map:?}"
        );
    }

    #[test]
    fn a_model_priced_on_one_dimension_only_stays_unpriced() {
        let (entries, skipped) =
            pricing_entries(&sample_catalogue(), "anthropic").expect("entries");
        let map = entries_as_map(&entries);
        assert!(
            !map.contains_key(&ModelId::new("input-only-preview")),
            "a model priced on input only must not gain an entry that charges $0 for output"
        );
        assert_eq!(
            skipped,
            vec!["input-only-preview".to_owned()],
            "the partially priced model is listed as skipped"
        );
        let rewritten = apply_pricing(
            "[provider]\nname = \"x\"\napi_key = \"k\"\nauth_style = \"bearer\"\n",
            &entries,
            "https://example.test",
        )
        .expect("rewrite");
        let parsed: ProviderFile = toml::from_str(&rewritten).expect("rewritten file parses");
        assert!(
            !parsed
                .provider
                .pricing
                .contains_key(&ModelId::new("input-only-preview")),
            "the rewritten TOML has no entry for the partially priced model: {rewritten}"
        );
    }

    #[test]
    fn an_unknown_models_dev_id_is_an_error_not_empty_pricing() {
        let error = pricing_entries(&sample_catalogue(), "ghost").expect_err("unknown id");
        assert!(error.to_string().contains("ghost"), "got {error:#}");
    }

    #[test]
    fn an_exponent_form_price_is_parsed_not_rejected() {
        // serde_json renders small floats in exponent form (shortest
        // round-trip); the plain-decimal parser rejects that text, so the
        // scientific parser must take over.
        let number: Value = serde_json::from_str("1e-6").expect("exponent json");
        assert_eq!(
            number.to_string(),
            "1e-6",
            "precondition: the catalogue number really renders in exponent form"
        );
        let per_token = super::per_token(Some(&number))
            .expect("an exponent-form price is an exact decimal")
            .expect("and the field was present");
        assert_eq!(per_token.normalize().to_string(), "0.000000000001");
    }

    #[test]
    fn exponent_form_prices_do_not_abort_the_providers_sync() {
        let cheap: Value = serde_json::from_str("1e-6").expect("exponent json");
        let catalogue = serde_json::json!({
            "tiny": {
                "models": {
                    "cheap-model": {
                        "cost": { "input": cheap, "output": 2.0 }
                    },
                    "plain-model": {
                        "cost": { "input": 3, "output": 4 }
                    }
                }
            }
        });
        let (entries, skipped) = pricing_entries(&catalogue, "tiny")
            .expect("the sync survives a price serde_json renders as 1e-6");
        let map = entries_as_map(&entries);
        assert!(
            skipped.is_empty(),
            "both models are fully priced, got {skipped:?}"
        );
        assert_eq!(map.len(), 2, "both models are priced, got {map:?}");
        let cheap = map
            .get(&ModelId::new("cheap-model"))
            .expect("the exponent-form model is priced");
        assert_eq!(cheap.input.normalize().to_string(), "0.000000000001");
        assert_eq!(cheap.output.normalize().to_string(), "0.000002");
    }

    #[test]
    fn a_non_numeric_catalogue_price_is_still_rejected() {
        let garbage = Value::String("free!".to_owned());
        let error = super::per_token(Some(&garbage)).expect_err("not a number");
        assert!(
            error.to_string().contains("not an exact decimal"),
            "got {error:#}"
        );
    }

    #[test]
    fn the_rewrite_replaces_pricing_and_keeps_the_rest_of_the_file() {
        let source = r#"
# A hand-written provider file with comments that must survive.
[provider]
name = "anthropic-direct"
api_key = "${ANTHROPIC_KEY}"
auth_style = "x-api-key"

[provider.adapters.anthropic]
protocol = "anthropic_messages"
endpoint = "https://api.anthropic.com/v1/messages"

[provider.pricing."claude-3-opus"]  # manual entry, replaced by the sync
input = "9.9"
output = "9.9"

[provider.pricing_sync]
models_dev_id = "anthropic"
"#;
        let entries = vec![(
            "claude-sonnet-4-5".to_owned(),
            ModelPricing {
                input: "0.000003".parse().expect("decimal"),
                output: "0.000015".parse().expect("decimal"),
                cache_creation: "0.00000375".parse().expect("decimal"),
                cache_read: "0.0000003".parse().expect("decimal"),
                reasoning: Decimal::ZERO,
            },
        )];
        let rewritten =
            apply_pricing(source, &entries, "https://example.test/api.json").expect("rewrite");
        assert!(
            rewritten.contains("hand-written provider file"),
            "comments outside pricing survive: {rewritten}"
        );

        let parsed: ProviderFile = toml::from_str(&rewritten).expect("rewritten file parses");
        let pricing = &parsed.provider.pricing;
        assert!(
            !pricing.contains_key(&ModelId::new("claude-3-opus")),
            "the sync replaces the whole pricing table"
        );
        let synced = pricing
            .get(&ModelId::new("claude-sonnet-4-5"))
            .expect("synced entry present");
        assert_eq!(synced.input.to_string(), "0.000003");
        assert_eq!(synced.cache_creation.to_string(), "0.00000375");
        assert_eq!(
            parsed
                .provider
                .pricing_sync
                .as_ref()
                .map(|s| s.models_dev_id.as_str()),
            Some("anthropic"),
            "the opt-in section survives the rewrite"
        );
        assert_eq!(
            parsed.provider.adapters.len(),
            1,
            "unrelated tables survive the rewrite"
        );
    }

    #[test]
    fn written_prices_round_trip_through_the_config_parser() {
        // Values written as strings must reparse as the same decimals —
        // the config layer rejects unquoted floats by design.
        let (entries, _) = pricing_entries(&sample_catalogue(), "anthropic").expect("entries");
        let rewritten = apply_pricing(
            "[provider]\nname = \"x\"\napi_key = \"k\"\nauth_style = \"bearer\"\n",
            &entries,
            "https://example.test",
        )
        .expect("rewrite");
        let parsed: ProviderFile = toml::from_str(&rewritten).expect("reparses");
        let synced = entries_as_map(&entries);
        for (model, pricing) in &parsed.provider.pricing {
            let expected = synced
                .get(model)
                .unwrap_or_else(|| panic!("unexpected model {model:?}"));
            assert_eq!(pricing.input, expected.input, "{model:?}");
            assert_eq!(pricing.output, expected.output, "{model:?}");
            assert_eq!(pricing.cache_read, expected.cache_read, "{model:?}");
            assert_eq!(pricing.cache_creation, expected.cache_creation, "{model:?}");
        }
    }
}
