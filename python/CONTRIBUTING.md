# Contributing

BeeAI Python is an open-source project committed to bringing LLM agents to people of all backgrounds. This page describes how you can join the BeeAI community in this goal.

## Before you start

If you are new to BeeAI contributing, we recommend you do the following before diving into the code:

- Read [Code of Conduct](/CODE_OF_CONDUCT.md).

## Style and lint

BeeAI Python uses the following tools to meet code quality standards and ensure a unified code style across the codebase:

We use the following libs to check the Python code:

- [Black](https://black.readthedocs.io/) - Code Formatter
- [Ruff](https://beta.ruff.rs/docs/) - Fast Python linter

Simple scripts for [Poetry](#poetry) are included to help you to review your changes and commit them.

## Issues and pull requests

We use GitHub pull requests to accept contributions.

While not required, opening a new issue about the bug you're fixing or the feature you're working on before you open a pull request is important in starting a discussion with the community about your work. The issue gives us a place to talk about the idea and how we can work together to implement it in the code. It also lets the community know what you're working on, and if you need help, you can reference the issue when discussing it with other community and team members.

If you've written some code but need help finishing it, want to get initial feedback on it before finishing it, or want to share it and discuss it prior to completing the implementation, you can open a Draft pull request and prepend the title with the [WIP] tag (for Work In Progress). This will indicate to reviewers that the code in the PR isn't in its final state and will change. It also means we will only merge the commit once it is finished. You or a reviewer can remove the [WIP] tag when the code is ready to be thoroughly reviewed for merging.

## Choose an issue to work on

BeeAI Python uses the following labels to help non-maintainers find issues best suited to their interests and experience level:

- [good first issue](https://github.com/i-am-bee/beeai-framework/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) - these issues are typically the simplest available to work on, ideal for newcomers. They should already be fully scoped, with a straightforward approach outlined in the descriptions.
- [help wanted](https://github.com/i-am-bee/beeai-framework/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) - these issues are generally more complex than good first issues. They typically cover work that core maintainers don't currently have the capacity to implement and may require more investigation/discussion. These are great options for experienced contributors looking for something more challenging.

## Setting up a local development environment

### Prerequisites

For development, there are some tools you will need prior cloning the code.

#### Python

We recommend using Python 3.11 or higher. First, ensure you have Python installed:

```bash
python --version
```

#### Poetry

[Poetry](https://python-poetry.org/) is a tool for Python packaging, dependency and virtual environment management that is used to manage the development of this project. Verify version two (V2) is installed on your machine. There are several ways to install it including through the package manager of your operating system, however, the easiest way to install is using the official installer, as follows:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

You can also use `pip` and `pipx` to install poetry.

Once you have Poetry installed, you will also need to add the poetry shell plugin:

```bash
poetry self add poetry-plugin-shell
```

> [!IMPORTANT]
> You must have poetry >= 2.0 installed

### Clone and set up the code

Follow these steps:

```bash
# Clone the repository
git clone https://github.com/i-am-bee/beeai-framework.git

# Enter the repository
cd beeai-framework

# Ensure you have the pre-commit hooks installed
.githooks/install.sh

# Enter the Python framework
cd python

# Use Poetry to install the project dependencies and activate a virtual environment
poetry install --all-extras
poetry shell

# Copy .env.example to .env and fill in required values
cp .env.example .env
```

#### Build the package:

```bash
poetry build
```

#### Test the Build Locally (Recommended)

Note: This should be be done outside an existing virtual environment or poetry shell.

```bash
# Create a virtual environment
python -m venv test_env

source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install the built package
pip install dist/beeai_framework-0.1.1.tar.gz
```

#### Run Linters/Formatters

Ensure your changes meet code quality standards:

- lint: use the next command run Black and Ruff:

```bash
poe lint # poe lint --fix
poe format # poe format --fix
```

#### Run Static Type Checker using mypy

MyPy has been integrated as a poe task in our development workflow. A set of comprehensive tasks are available for different type checking scenarios.

Running the basic type check looks for all python files in current directory and subdirectories.

```bash
poe type-check
```
Support for dynamic flags is available via the `${POE_ARGS}`. The `${POE_ARGS}` allows passing flags like --strict or --verbose.

Running with the `--strict` flag enables strict type checking mode.

```bash
poe type-check --strict
```

Running with the `--verbose` flag offers more detailed messages during type checking.

```bash
poe type-check --verbose
```

#### Run Tests

Ensure your changes pass all tests:

```bash
# Run Unit tests
poe test --type unit

# Run integration tests
poe test --type integration

# Run E2E tests
poe test --type e2e
```

### Other useful commands

- `poe build`
- `poe commit`
- `poe copyright` and `poe copyright --type check`
- `poe docs --type build` and `poe docs --type check`

#### Follow Conventional Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) to structure our commit messages. Please use the following format:

```
<type>(<scope>): <subject>
```

- Type: feat, fix, chore, docs, style, refactor, perf, test, etc.
- Scope: The area of the codebase your changes affect (optional). The allowed values are: adapters, agents, llms, tools, cache, emitter, internals, logger, memory, serializer, infra, deps, instrumentation
- Subject: A short description of the changes (required)

_Example:_

```
feat(llm): add streaming support for watsonx adapter

Ref: #15
```

#### Commit:

- commit: for convenience you can use the following command to sign-off your commit with `-s` and generate the commit.

```bash
poe commit "<type>(<scope>): <subject>"
```

By following these steps, you'll be all set to contribute to our project! If you encounter any issues during the setup process, please feel free to open an issue.

## Updating examples and embedding

Currently [embedme](https://github.com/zakhenry/embedme) is used to embed code examples directly in documentation. Supported file types can be found [here](https://github.com/zakhenry/embedme?tab=readme-ov-file#multi-language).

Once an example is edited or a new one is created and referenced running the following command will update the documentation.

```bash
poe docs --type build
```

## Legal

The following sections detail important legal information that should be viewed prior to contribution.

### License and Copyright

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: [Apache-2.0](https://spdx.org/licenses/Apache-2.0)

If you would like to see the detailed LICENSE click [here](/LICENSE).

### Developer Certificate of Origin (DCO)

We have tried to make it as easy as possible to make contributions. This applies to how we handle the legal aspects of contribution. We use the same approach - the [Developer's Certificate of Origin 1.1 (DCO)](https://developercertificate.org/) - that the Linux® Kernel [community](https://docs.kernel.org/process/submitting-patches.html#sign-your-work-the-developer-s-certificate-of-origin) uses to manage code contributions.

We ask that when submitting a patch for review, the developer must include a sign-off statement in the commit message. If you set your `user.name` and `user.email` in your `git config` file, you can sign your commit automatically by using the following command:

```bash
git commit -s
```

If a commit has already been created but signoff was missed this can be remedied

```bash
git --amend -s
```

The following example includes a `Signed-off-by:` line, which indicates that the submitter has accepted the DCO:

```txt
Signed-off-by: John Doe <john.doe@example.com>
```

We automatically verify that all commit messages contain a `Signed-off-by:` line with your email address.

#### Useful tools for doing DCO signoffs

While the web ui natively supports this now, there are a number of tools that make it easier for developers to manage DCO signoffs if not using the web interface.

- DCO command line tool, which lets you do a single signoff for an entire repo ( <https://github.com/coderanger/dco> )
- GitHub UI integrations for adding the signoff automatically ( <https://github.com/scottrigby/dco-gh-ui> )
- Chrome - <https://chrome.google.com/webstore/detail/dco-github-ui/onhgmjhnaeipfgacbglaphlmllkpoijo>
- Firefox - <https://addons.mozilla.org/en-US/firefox/addon/scott-rigby/?src=search>
