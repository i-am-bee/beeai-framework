# Feedo Unit Test Results

This document contains the execution logs for the fully mocked unit tests verifying the `FeedoSearchTool` logic and input schema validations across both Python and TypeScript environments.

## Python Test Run (`pytest`)

**Command:**
```bash
pytest tests/tools/test_feedo.py -v
```

**Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.0, pluggy-1.6.0 -- C:\Users\andri\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
rootdir: D:\Projects\Development\Projects\feedo\beeai-framework\python
configfile: pyproject.toml
plugins: anyio-4.12.1, asyncio-1.4.0, respx-0.23.1
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... 
collected 5 items

tests/tools/test_feedo.py::test_feedo_add PASSED                        [ 20%]
tests/tools/test_feedo.py::test_feedo_search PASSED                     [ 40%]
tests/tools/test_feedo.py::test_feedo_search_empty PASSED               [ 60%]
tests/tools/test_feedo.py::test_feedo_update PASSED                     [ 80%]
tests/tools/test_feedo.py::test_feedo_delete PASSED                     [100%]

============================== 5 passed in 0.48s ==============================
```

## TypeScript Test Run (`vitest`)

**Command:**
```bash
npx vitest run src/tools/search/feedo.test.ts
```

**Output:**
```
 RUN  v2.1.9 D:/Projects/Development/Projects/feedo/beeai-framework/typescript

 ✓ src/tools/search/feedo.test.ts (4 tests) 9ms
   ✓ FeedoSearchTool > should successfully perform 'add' action
   ✓ FeedoSearchTool > should successfully perform 'search' action
   ✓ FeedoSearchTool > should successfully perform 'update' action
   ✓ FeedoSearchTool > should successfully perform 'delete' action

 Test Files  1 passed (1)
      Tests  4 passed (4)
   Start at  14:31:21
   Duration  857ms (transform 221ms, setup 139ms, collect 390ms, tests 9ms, environment 0ms, prepare 117ms)
```
