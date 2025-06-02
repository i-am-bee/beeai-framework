# 🔍 SearXNGTool

## Description

This tool allows an agent to run web searches using SearXNG. SearXNG is a metasearch engine that aggregates results from multiple search engines. SearXNG does not require an API key, you can run SearXNG directly on your laptop to faciliate easy access to web search functionality for your agent.

## Setup

Follow the steps to create a private SearXNG instance. For more advanced usage see the [SearXNG project documentation](https://github.com/searxng/searxng).

### 1. Create a local folder for the SearXNG configuration files.

The files will be automatically written to this location, but you will need to make a minor modification.

```
 mkdir ~/searxng
```

### 2. Run the SearXNG docker container.

```
docker run -d --name searxng -p 8888:8080 -v ./searxng:/etc/searxng --restart always searxng/searxng:latest
```

### 3. Edit the configuration files and restart the container.

When you first run a SearXNG docker container, it will write configuration files to the `~/searxng` folder.

```
settings.yml
uwsgi.ini
```

Open `~/searxng/settings.yml`, find the `formats:` list and add `json`.

```yaml
search:
  formats:
    - html
    - json
```

Stop and restart the container.

### 4. Check installation

Navigate to `http://localhost:8888` and you should see an SearXNG interface.

## Usage

Configure the SEARXNG_BASE_URL environment variable:

```
SEARXNG_BASE_URL="http://127.0.0.1:8888"
```

Run the tool:

```js
import "dotenv/config";
import { SearXNGTool } from "beeai-framework/tools/search/searXNGSearch";

const tool = new SearXNGTool();
const result = await tool.run({
  query: "Worlds longest lived vertebrate",
});

console.log(result.getTextContent());
```
