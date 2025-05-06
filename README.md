# TianGong AI LangGraph Server

## Install dependencies

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash

nvm install
nvm alias default 20
nvm use

npm install
# Update npm packages
npm update && npm ci
```

## Start server

```bash

# Windows & Linux
sudo pip install -U --break-system-packages langgraph-cli

# Mac Only
brew install langgraph-cli

sudo langgraph up
```

```bash
nohup node dist/multi_agents/kg_textbooks.js > kg_textbook.log 2>&1 &
tmux new -d -s neo4j_import 'node dist/multi_agents/kg_textbooks.js > kg_textbook.log 2>&1'
tmux kill-session -t neo4j_import
```

## Test prototype

```bash
npx ts-node src/prototype/structured_output.ts
```
