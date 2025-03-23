import { Client } from '@langchain/langgraph-sdk';
import { RemoteGraph } from '@langchain/langgraph/remote';
import dotenv from 'dotenv';
import fs from 'fs';
import PQueue from 'p-queue';

dotenv.config();

const url = process.env.REMOTE_DEPLOYMENT_URL ?? '';
const apiKey = process.env.LANGSMITH_API_KEY ?? '';
const graphName = process.env.GRAPH_DATA_SYNTHESIZE ?? '';
const client = new Client({ apiUrl: url, apiKey: apiKey });
const remoteGraph = new RemoteGraph({ graphId: graphName, url: url, apiKey: apiKey });

type questionElement = {
  index: number;
  question: string;
  answer: string;
  difficulty: string;
  type: string;
  field: string;
};

type dataElement = {
  index: number;
  question: string;
  resposne: string;
  chain_of_thought: string;
};

async function synthesizeData(question: string) {
  const thread = await client.threads.create();
  const config = { configurable: { thread_id: thread.thread_id } };
  const response = await remoteGraph.invoke({ input: question }, config);
  return response;
}

const dataset = 'question';
const input_file = './src/data/kiln/input/' + dataset + '.json';
const output_file = './src/data/kiln/output/' + dataset + '.json';

async function main() {
  const data: questionElement[] = JSON.parse(fs.readFileSync(input_file, 'utf-8'));
  const queue = new PQueue({ concurrency: 5 });
  const results: dataElement[] = [];

  const tasks = data.map((item) =>
    queue.add(async () => {
      console.log(`Processing: ${item.question}`);
      try {
        const response = await synthesizeData(item.question);
        const data: dataElement[] = response.chunk_analysis;
        data.forEach((element) => {
          element.index = item.index;
          results.push(element);
        });
      } catch (error) {
        console.error(`Error processing question: ${item.index}`, error);
      }
    }),
  );

  await Promise.all(tasks);

  fs.writeFileSync(output_file, JSON.stringify(results, null, 2), 'utf-8');
  console.info('All tasks completed.');
}

main().catch(console.error);
