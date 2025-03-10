# 🛢️ SQLTool

## Description

This tool executes SQL queries and guides the agent in constructing correct queries based on the structures of all available tables in the database. It simplifies data retrieval, allowing users without advanced SQL knowledge to effectively query databases using natural language. The tool allows only `SELECT` queries to prevent any accidental changes to the database.

## Requirements

To use SQLTool with different databases, ensure you have the following:

- **Sequelize**: Version 6
- **Database Connector Package**: The appropriate package for your database (e.g., `ibm_db`, `mysql2`, `sqlite3`, etc.)

## Installation

Follow the table below to install the required packages for your database:

| Database             | Required Package(s)   |
| -------------------- | --------------------- |
| IBM Db2 for LUW      | `node-gyp`, `ibm_db`  |
| SQLite               | `sqlite3`             |
| MariaDB              | `mariadb` (version 2) |
| MySQL                | `mysql2`              |
| PostgreSQL           | `pg`                  |
| Microsoft SQL Server | `tedious`             |
| Oracle               | `oracledb`            |

To install Sequelize, run the following command:

```bash
yarn add sequelize
```

Depending on the database you're using, install the required package(s). For example, if you're using IBM Db2 for LUW, run:

```bash
yarn add node-gyp ibm_db
```

Replace `node-gyp` `ibm_db` with the appropriate package(s) for your database from the table above.

## Usage

To use the `SQLTool` class, the following parameters must be supplied:

- `provider`: The database provider. Supported values are: `mysql`, `mariadb`, `postgres`, `mssql`, `db2`, `sqlite`, `oracle`.

- `connection`: This parameter is based on the [Sequelize](https://sequelize.org/api/v6/class/src/sequelize.js~sequelize#instance-constructor-constructor) `Options` type, which specifies the required configuration for establishing a database connection in Sequelize. You can use this object to pass various connection parameters required by the database [dialect](https://sequelize.org/docs/v6/other-topics/dialect-specific-things/).

**For SQLite**

```js
import { SQLTool } from "beeai-framework/tools/database/sql.js";

const sqlTool = new SQLTool({
  provider: "sqlite",
  connection: {
    dialect: "sqlite",
    storage: "sqlite_file.db",
    logging: false,
  },
});
```

**Other databases**

```js
import { SQLTool } from "beeai-framework/tools/database/sql.js";

const sqlTool = new SQLTool({
  provider: "oracle", // one of "mariadb" | "mysql" | "mssql" | "postgres" | "db2"
  connection: {
    dialect: "oracle", // one of "mariadb" | "mysql" | "mssql" | "postgres" | "db2"
    host: "localhost",
    port: 1521,
    database: "database_name",
    username: "username",
    password: "password",
    schema: "schema_name", // Schema is optional for mariadb, mysql, mssql, postgres
    logging: false,
  },
});
```

## Examples

Below are examples showcasing how to perform a search on the [chinook](https://github.com/lerocha/chinook-database) SQLite database:

**_Question 1_**

_Which country's customers spent the most?_

```
Agent (thought) 🤖 :  I need to access the database to find the answer, and I have a SQLTool that can help me with that.
Agent (tool_name) 🤖 :  SQLTool
Agent (tool_input) 🤖 :  {"query": "SELECT country, SUM(amount) as total_spent FROM customer_spending GROUP BY country ORDER BY total_spent DESC LIMIT 1"}
Agent (tool_output) 🤖 :  {"error":"Based on this database schema structure: Table 'album' with columns: album_id (INTEGER), title (NVARCHAR(160)), artist_id (INTEGER); Table 'artist' with columns: artist_id (INTEGER), name (NVARCHAR(120)); Table 'customer' with columns: customer_id (INTEGER), first_name (NVARCHAR(40)), last_name (NVARCHAR(20)), company (NVARCHAR(80)), address (NVARCHAR(70)), city (NVARCHAR(40)), state (NVARCHAR(40)), country (NVARCHAR(40)), postal_code (NVARCHAR(10)), phone (NVARCHAR(24)), fax (NVARCHAR(24)), email (NVARCHAR(60)), support_rep_id (INTEGER); Table 'employee' with columns: employee_id (INTEGER), last_name (NVARCHAR(20)), first_name (NVARCHAR(20)), title (NVARCHAR(30)), reports_to (INTEGER), birthdate (DATETIME), hire_date (DATETIME), address (NVARCHAR(70)), city (NVARCHAR(40)), state (NVARCHAR(40)), country (NVARCHAR(40)), postal_code (NVARCHAR(10)), phone (NVARCHAR(24)), fax (NVARCHAR(24)), email (NVARCHAR(60)); Table 'genre' with columns: genre_id (INTEGER), name (NVARCHAR(120)); Table 'invoice' with columns: invoice_id (INTEGER), customer_id (INTEGER), invoice_date (DATETIME), billing_address (NVARCHAR(70)), billing_city (NVARCHAR(40)), billing_state (NVARCHAR(40)), billing_country (NVARCHAR(40)), billing_postal_code (NVARCHAR(10)), total (NUMERIC(10,2)); Table 'invoice_line' with columns: invoice_line_id (INTEGER), invoice_id (INTEGER), track_id (INTEGER), unit_price (NUMERIC(10,2)), quantity (INTEGER); Table 'media_type' with columns: media_type_id (INTEGER), name (NVARCHAR(120)); Table 'playlist' with columns: playlist_id (INTEGER), name (NVARCHAR(120)); Table 'playlist_track' with columns: playlist_id (INTEGER), track_id (INTEGER); Table 'track' with columns: track_id (INTEGER), name (NVARCHAR(200)), album_id (INTEGER), media_type_id (INTEGER), genre_id (INTEGER), composer (NVARCHAR(220)), milliseconds (INTEGER), bytes (INTEGER), unit_price (NUMERIC(10,2)), \n      generate a correct query that retrieves data using the appropriate sqlite dialect. \n      The original request was: SELECT country, SUM(amount) as total_spent FROM customer_spending GROUP BY country ORDER BY total_spent DESC LIMIT 1, and the error was: SQLITE_ERROR: no such table: customer_spending."}
```

On the first iteration, the agent tried to guess the SQL query based on the question, resulting in a missing table error. However, by utilizing the metadata returned with the error, it self-corrected on the second iteration and successfully generated the correct query.

```
Agent (thought) 🤖 :  It seems that the SQLTool didn't work as expected, and I need to adjust the query to match the available tables in the database.
Agent (tool_name) 🤖 :  SQLTool
Agent (tool_input) 🤖 :  {"query":"SELECT i.billing_country, SUM(i.total) as total_spent FROM invoice i GROUP BY i.billing_country ORDER BY total_spent DESC LIMIT 1"}
Agent (tool_output) 🤖 :  {"success":true,"results":[{"billing_country":"USA","total_spent":1040.49}]}
Agent (thought) 🤖 :  The SQLTool was successful this time, and I have the answer to the question.
Agent (final_answer) 🤖 :  The country whose customers spent the most is the USA, with a total spending of $1040.49.
```

**_Generated SQL_**

```sql
SELECT country,
       SUM(amount) AS total_spent
  FROM customer_spending
 GROUP BY country
 ORDER BY total_spent DESC
 LIMIT 1;
```

**_Answer_**

_The country whose customers spent the most is the USA, with a total spending of $1040.49._

**_Question 2_**

_Show the top 3 best selling artists in terms of revenue._

**_Generated SQL_**

```sql
SELECT a.name,
       SUM(il.unit_price * il.quantity) AS total_revenue
  FROM invoice_line il
       JOIN
       track t ON il.track_id = t.track_id
       JOIN
       album al ON t.album_id = al.album_id
       JOIN
       artist a ON al.artist_id = a.artist_id
 GROUP BY a.name
 ORDER BY total_revenue DESC
 LIMIT 3;
```

**_Answer_**

_The top 3 best selling artists in terms of revenue are:_
_- Queen with a total revenue of \$190.08_
_- Jimi Hendrix with a total revenue of \$185.13_
_- Red Hot Chili Peppers with a total revenue of \$128.77._

## Sample Agent

A complete sample of an SQL agent is available [here](/typescript/examples/agents/sql.ts).
