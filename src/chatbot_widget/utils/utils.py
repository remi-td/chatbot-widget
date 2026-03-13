import asyncio
import nest_asyncio

nest_asyncio.apply()

def run_async(coro):
    """
    Run an async coroutine safely in both normal Python scripts and Jupyter notebooks.

    This utility ensures compatibility with environments that already have an active
    asyncio event loop (e.g. Jupyter, IPython) where `asyncio.run()` would otherwise fail.

    Parameters
    ----------
    coro : coroutine
        The async coroutine object to run (e.g. `my_async_func()`).

    Returns
    -------
    Any
        The result returned by the coroutine after it completes execution.

    Rationale
    ---------
    - In standard Python scripts, you can safely call `asyncio.run(coro)`.
    - In Jupyter or IPython, an event loop is already running — calling `asyncio.run()`
      there raises `RuntimeError: asyncio.run() cannot be called from a running event loop`.
    - This helper catches that case, applies `nest_asyncio`, and reuses the existing loop
      via `loop.run_until_complete()`, allowing the coroutine to execute normally.

    Examples
    --------
    >>> async def fetch_data():
    ...     await asyncio.sleep(1)
    ...     return "done"

    # Works in both notebooks and scripts:
    >>> result = run_async(fetch_data())
    >>> print(result)
    done

    # In a class method:
    >>> class MyClient:
    ...     async def get_data(self): ...
    ...     def setup(self):
    ...         data = run_async(self.get_data())

    Do's and Don'ts
    ---------------
    ✅ **Do** use this when you want to call async functions from synchronous code,
       especially inside Jupyter or non-async class methods.

    ✅ **Do** call it with a coroutine object (e.g. `my_async_func()`), not the function itself.

    ❌ **Don't** use it *inside* another async function — just `await` directly there instead.

    ❌ **Don''t** wrap blocking or CPU-heavy code in async functions unless truly needed.

    Notes
    -----
    - Applying `nest_asyncio` once per process is harmless; it simply patches asyncio
      to allow nested event loops.
    - This function is a practical workaround when you want sync-like control flow
      without refactoring your entire codebase to be async.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)
