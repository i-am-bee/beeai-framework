## Set up a development environment

This project uses [Mise-en-place](https://mise.jdx.dev/) as a manager of tool versions (`python`, `poetry`, `nodejs`, `yarn` etc.), as well as a task runner and environment manager. Mise will download all the needed tools automatically -- you don't need to install them yourself.

Clone this project, then run these setup steps:

```sh
curl https://mise.run | sh # more ways to install: https://mise.jdx.dev/installing-mise.html
mise trust
mise install
```

After setup, you can use:

- `mise run` to list tasks and select one interactively to run

- `mise <task-name>` to run a task

- `mise x -- <command>` to run a project tool -- for example `mise x -- poetry add <package>`

If you want to run tools directly without the `mise x --` prefix, you need to activate a shell hook:

- Bash: `eval "$(mise activate bash)"` (add to `~/.bashrc` to make permanent)

- Zsh: `eval "$(mise activate zsh)"` (add to `~/.zshrc` to make permanent)

- Fish: `mise activate fish | source` (add to `~/.config/fish/config.fish` to make permanent)

- Other shells: [documentation](https://mise.jdx.dev/installing-mise.html#shells)

Some tasks to get you started:

- `mise docs:check` to run formatters and linters (also runs on commit and in CI)
- `mise docs:fix` to embed snippets and fix issues where possible
- `mise docs:run` to view docs in the browser
