use std::future::Future;
use std::pin::Pin;

/// A future that is `Send` and `'static`.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// Controls whether CPU-bound operations run sequentially or in parallel.
///
/// - [`Executor::sequential()`] — default; runs everything inline, no extra threads.
/// - [`Executor::parallel()`]   — uses rayon for `par_map` and
///   `tokio::task::spawn_blocking` for blocking work.
#[derive(Debug, Clone, Copy, Default)]
pub struct Executor {
    parallel: bool,
}

impl Executor {
    /// Returns an executor that runs all work inline on the calling thread.
    pub fn sequential() -> Self {
        Self { parallel: false }
    }

    /// Returns an executor that parallelises work using rayon and `spawn_blocking`.
    pub fn parallel() -> Self {
        Self { parallel: true }
    }

    /// Map `f` over `items`, in parallel when the executor is configured to do so.
    pub fn par_map<I, O, F>(&self, items: Vec<I>, f: F) -> Vec<O>
    where
        I: Send,
        O: Send,
        F: Fn(I) -> O + Send + Sync,
    {
        if self.parallel {
            use rayon::prelude::*;
            items.into_par_iter().map(f).collect()
        } else {
            items.into_iter().map(f).collect()
        }
    }

    /// Run a blocking closure, offloading to `tokio::task::spawn_blocking` when
    /// the executor is configured for parallel execution.
    pub fn spawn_blocking<F, T>(&self, f: F) -> BoxFuture<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        if self.parallel {
            Box::pin(async move {
                tokio::task::spawn_blocking(f)
                    .await
                    .expect("blocking task panicked")
            })
        } else {
            Box::pin(std::future::ready(f()))
        }
    }
}
