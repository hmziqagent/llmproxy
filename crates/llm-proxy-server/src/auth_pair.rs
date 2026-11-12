//! The live (authenticator, admission rules) pair: one atomically-replaced
//! unit behind a single lock, so no instant exists where the router
//! authenticates against one key set while the engine enforces another's
//! rules.

use std::sync::Arc;

use llm_proxy_engine::Admission;

use crate::middleware::ClientAuth;

/// One generation of what a request is served by: the authenticator the
/// router's auth check reads and the admission rules the engine enforces,
/// fetched from one `Arc` so they can never disagree about which generation
/// is installed.
#[derive(Debug)]
pub(crate) struct AuthPair {
    /// The authenticator half: verifiers plus the per-`ClientAuth` digest
    /// cache.
    client_auth: Arc<ClientAuth>,
    /// The rules half: allowlists, windows, budgets and caps.
    admission: Arc<dyn Admission>,
}

impl AuthPair {
    /// The authenticator serving in this generation.
    pub(crate) fn client_auth(&self) -> &Arc<ClientAuth> {
        &self.client_auth
    }

    /// The admission rules enforcing in this generation.
    pub(crate) fn admission(&self) -> &Arc<dyn Admission> {
        &self.admission
    }
}

/// The authenticator and the admission rules as ONE atomically-replaced
/// pair: the router's auth check reads the pair's client-auth half per
/// request, and the engine holds this same object as its
/// [`Admission`](llm_proxy_engine::Admission) step (installed through
/// [`AppState::with_auth_pair`](crate::AppState::with_auth_pair)), so a
/// runtime key change swaps both halves under one write guard. A revoked
/// token therefore stops authenticating in the same instant the new rules
/// forget its id — never one statement apart, where the old token could
/// still authenticate while already being unknown (and thus unrestricted)
/// to the new rules, nor the mirror, where a just-issued key's limits had
/// not loaded yet.
///
/// A swap may also replace one half alone ([`Self::replace_admission`],
/// [`Self::replace_client_auth`]): the pair still moves from one
/// self-consistent generation to the next under the same single write
/// guard; no mixed pair is ever installed.
///
/// The lock is never held across an `.await`: each method clones the
/// current pair (or its half) out and delegates to the clone. Reservations
/// settle on whichever rules are current when they settle, so every
/// admission installed for one engine must share the same underlying
/// backend; charges and cooldowns then survive the swap.
///
/// The per-`ClientAuth` digest cache dies with each swap by design:
/// preserving it would let a revoked token's cached digest survive the
/// swap that revoked it, so one extra argon2 per live client per refresh is
/// the price of correct revocation.
pub struct SwappingAuthPair {
    current: std::sync::RwLock<Arc<AuthPair>>,
}

impl SwappingAuthPair {
    /// A pair serving `client_auth` for authentication and `admission` for
    /// enforcement until [`Self::replace`] swaps both.
    #[must_use]
    pub fn new(client_auth: Arc<ClientAuth>, admission: Arc<dyn Admission>) -> Self {
        Self {
            current: std::sync::RwLock::new(Arc::new(AuthPair {
                client_auth,
                admission,
            })),
        }
    }

    /// The pair serving right now: one read, so both halves come from the
    /// same generation.
    pub(crate) fn current(&self) -> Arc<AuthPair> {
        let guard = self
            .current
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Arc::clone(&guard)
    }

    /// The authenticator serving right now — the read the router's auth
    /// check makes per request.
    pub(crate) fn current_client_auth(&self) -> Arc<ClientAuth> {
        Arc::clone(self.current().client_auth())
    }

    /// Atomically replace the authenticator AND the admission rules — one
    /// write guard, so no served instant mixes the old half with the new.
    pub fn replace(&self, client_auth: Arc<ClientAuth>, admission: Arc<dyn Admission>) {
        let mut guard = self
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = Arc::new(AuthPair {
            client_auth,
            admission,
        });
    }

    /// Replace the admission rules, keeping the serving authenticator —
    /// still one write guard, so the pair moves between self-consistent
    /// generations.
    pub fn replace_admission(&self, admission: Arc<dyn Admission>) {
        let mut guard = self
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let client_auth = Arc::clone(guard.client_auth());
        *guard = Arc::new(AuthPair {
            client_auth,
            admission,
        });
    }

    /// Replace the authenticator, keeping the enforced rules — still one
    /// write guard, so the pair moves between self-consistent generations
    /// (and the revoked token's cached digest dies with the old half).
    /// Returns the authenticator that was serving until now.
    pub(crate) fn replace_client_auth(&self, next: Arc<ClientAuth>) -> Arc<ClientAuth> {
        let mut guard = self
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = Arc::clone(guard.client_auth());
        let admission = Arc::clone(guard.admission());
        *guard = Arc::new(AuthPair {
            client_auth: next,
            admission,
        });
        previous
    }
}

impl std::fmt::Debug for SwappingAuthPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let current = self.current();
        f.debug_struct("SwappingAuthPair")
            .field("client_auth", current.client_auth())
            .field("admission", current.admission())
            .finish()
    }
}

#[async_trait::async_trait]
impl Admission for SwappingAuthPair {
    async fn authorize(
        &self,
        claim: &llm_proxy_engine::AdmissionClaim<'_>,
    ) -> Result<(), llm_proxy_engine::EngineError> {
        Arc::clone(self.current().admission())
            .authorize(claim)
            .await
    }

    async fn reserve(
        &self,
        claim: &llm_proxy_engine::AdmissionClaim<'_>,
    ) -> Result<llm_proxy_engine::AdmissionTicket, llm_proxy_engine::EngineError> {
        Arc::clone(self.current().admission()).reserve(claim).await
    }

    fn settle(
        &self,
        ticket: &llm_proxy_engine::AdmissionTicket,
        outcome: &llm_proxy_engine::AdmissionOutcome<'_>,
    ) {
        self.current().admission().settle(ticket, outcome);
    }

    async fn rekey(
        &self,
        ticket: &llm_proxy_engine::AdmissionTicket,
        member: &str,
        model: &str,
    ) -> Result<(), llm_proxy_engine::EngineError> {
        Arc::clone(self.current().admission())
            .rekey(ticket, member, model)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// One generation, held by identity: `current()` must return exactly
    /// these two halves together or the installed pair was mixed.
    #[derive(Debug)]
    struct Generation {
        client_auth: Arc<ClientAuth>,
        admission: Arc<dyn Admission>,
    }

    fn generation() -> Generation {
        Generation {
            client_auth: Arc::new(ClientAuth::fail_closed()),
            admission: Arc::new(llm_proxy_engine::AllowAllAdmission),
        }
    }

    /// After each swap the serving snapshot IS the installed pair: both
    /// halves by pointer identity, never a mixture of two generations.
    #[test]
    fn every_swap_installs_both_halves_together() {
        let first = generation();
        let live =
            SwappingAuthPair::new(Arc::clone(&first.client_auth), Arc::clone(&first.admission));
        for _ in 0..8 {
            let next = generation();
            live.replace(Arc::clone(&next.client_auth), Arc::clone(&next.admission));
            let snapshot = live.current();
            assert!(
                Arc::ptr_eq(snapshot.client_auth(), &next.client_auth)
                    && Arc::ptr_eq(snapshot.admission(), &next.admission),
                "the serving pair must be exactly the one just installed"
            );
        }
    }

    /// The structural invariant under fire: while a writer swaps whole
    /// pairs, every snapshot a reader takes matches some installed
    /// generation on BOTH halves. Two adjacent half-swaps (the design this
    /// type replaces) install a mixed pair between them — one half from
    /// the old generation, one from the new — which no generation matches,
    /// so this test fails against that regression.
    #[test]
    fn swaps_under_fire_never_serve_a_mixed_pair() {
        const SWAPS: usize = 4_096;
        let first = generation();
        let live = Arc::new(SwappingAuthPair::new(
            Arc::clone(&first.client_auth),
            Arc::clone(&first.admission),
        ));
        // Recorded before each swap; the record itself keeps every Arc
        // alive, so addresses never recycle into a false match.
        let installed: Arc<Mutex<Vec<Generation>>> = Arc::new(Mutex::new(vec![first]));

        let writer_live = Arc::clone(&live);
        let writer_installed = Arc::clone(&installed);
        let writer = std::thread::spawn(move || {
            for _ in 0..SWAPS {
                let next = generation();
                writer_installed
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(Generation {
                        client_auth: Arc::clone(&next.client_auth),
                        admission: Arc::clone(&next.admission),
                    });
                writer_live.replace(next.client_auth, next.admission);
            }
        });

        let mut snapshots = 0_usize;
        while snapshots == 0 || !writer.is_finished() {
            let snapshot = live.current();
            let installed = installed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let matched = installed.iter().any(|installed_generation| {
                Arc::ptr_eq(snapshot.client_auth(), &installed_generation.client_auth)
                    && Arc::ptr_eq(snapshot.admission(), &installed_generation.admission)
            });
            assert!(
                matched,
                "a snapshot mixed halves from two generations was served"
            );
            snapshots += 1;
        }
        writer.join().expect("writer finishes");
        assert!(
            snapshots > 0,
            "the reader must have observed the swaps at least once"
        );
    }
}
