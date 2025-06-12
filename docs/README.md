# BeeAI Framework Documention

## Set up a development environment

To start contributing to the BeeAI Framework Documentation, follow these steps to set up your development environment:

1.  **Install Node Version Manager (NVM):** We use `.nvmrc` to specify the required Node.js version. Install [nvm](https://github.com/nvm-sh/nvm) by following the official installation instructions.

2.  **Install the Correct Node.js Version:** Use `nvm` to install and use the Node.js version specified in the `.nvmrc` file:

```bash
nvm install
nvm use
```

3. **Install [Yarn](https://yarnpkg.com/) via Corepack:** This project uses Yarn as the package manager. Ensure you have Corepack enabled and install Yarn:

```bash
corepack enable
```

4.  **Install Dependencies:** Install all project dependencies by running:

```bash
yarn install --immutable
yarn prepare
```

5. **Run**

```bash
yarn dev
```

6. **Embed scripts**

```
yarn snippets:embed
```
