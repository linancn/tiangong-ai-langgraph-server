import { Client } from '@langchain/langgraph-sdk';
import { RemoteGraph } from '@langchain/langgraph/remote';
import dotenv from 'dotenv';
import fs from 'fs';
import PQueue from 'p-queue';

dotenv.config();

const url = process.env.LOCAL_DEPLOYMENT_URL ?? '';
const apiKey = process.env.LANGSMITH_API_KEY ?? '';
const graphName = process.env.GRAPH_GET_RESPONSE ?? '';
const client = new Client({ apiUrl: url, apiKey: apiKey });
const remoteGraph = new RemoteGraph({ graphId: graphName, url: url, apiKey: apiKey });

type questionElement = {
  index?: number;
  question: string;
  answer: string;
  difficulty: string;
  type: string;
  field: string;
  model?: string;
  response?: string;
};

// customize model name
const model_name = '';

async function getResponse(input: string) {
  const thread = await client.threads.create();
  const config = { configurable: { thread_id: thread.thread_id } };
  const response = await remoteGraph.invoke({
    messages: [{ role: 'human', content: input }],
    config,
  });

  return response;
}

async function main() {
  const data: questionElement[] = JSON.parse(fs.readFileSync('./src/data/question.json', 'utf-8'));
  const queue = new PQueue({ concurrency: 5 });
  const results: any[] = [];

  const tasks = data.map((item) =>
    queue.add(async () => {
      console.log(`Processing: ${item.question}`);
      try {
        const response = await getResponse(item.question);
        const answer = response.answers[response.answers.length - 1];
        item.model = model_name;
        item.response = answer;
        results.push(item);
      } catch (error) {
        console.error(`Error processing question: ${item.question}`, error);
      }
    }),
  );

  await Promise.all(tasks);

  fs.writeFileSync(
    './src/data/responses/' + model_name + '.json',
    JSON.stringify(results, null, 2),
    'utf-8',
  );
  console.info('All tasks completed.');
}

main().catch(console.error);
