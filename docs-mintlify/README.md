# BeeAI Framework Documention

Powered by Mintlify

### Development

Install the [Mintlify CLI](https://www.npmjs.com/package/mintlify) to preview the documentation changes locally. To install, use the following command

```
npm i -g mintlify
```

Run the following command from the documentation directory (i.e., where docs.json is located)

```
mintlify dev
```

### Publishing Changes

Install the Mintlify Github App to auto propagate changes from the repo to the deployment. Changes will be deployed to production automatically after pushing to the default branch. Find the link to install the app in org's Mintlify dashboard. 

#### Troubleshooting

- Mintlify dev isn't running - Run `mintlify install` it'll re-install dependencies.
- Page loads as a 404 - Make sure you are running in a folder with `docs.json`
