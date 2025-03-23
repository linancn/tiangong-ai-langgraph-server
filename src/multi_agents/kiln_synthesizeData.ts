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
  response: string;
  chain_of_thought: string;
};

async function synthesizeData(
  question: string,
  supabase_email: string,
  supabase_password: string,
  supabase_authorization: string,
) {
  const thread = await client.threads.create();
  const config = { configurable: { thread_id: thread.thread_id } };
  const response = await remoteGraph.invoke(
    {
      input: question,
      supabase_email: supabase_email,
      supabase_password: supabase_password,
      supabase_authorization: supabase_authorization,
    },
    config,
  );
  return response;
}

// Dataset to process
const dataset = 'question_test';
const input_file = './src/data/kiln/input/' + dataset + '.json';
const output_file = './src/data/kiln/output/' + dataset + '.json';

// supabase credentials
const supabase_email = process.env.EMAIL ?? '';
const supabase_password = process.env.PASSWORD ?? '';
const supabase_authorization = process.env.SUPABASE_ANON_KEY ?? '';

async function main() {
  const data: questionElement[] = JSON.parse(fs.readFileSync(input_file, 'utf-8'));
  const queue = new PQueue({ concurrency: 5 });
  const results: dataElement[] = [];
  // Initialize output file with empty array
  fs.writeFileSync(output_file, JSON.stringify([], null, 2), 'utf-8');

  // Create a separate queue for writing to file to avoid race conditions
  const writeQueue = new PQueue({ concurrency: 1 });

  const tasks = data.map((item) =>
    queue.add(async () => {
      console.log(`Processing: ${item.index}`);
      try {
        const response = await synthesizeData(
          item.question,
          supabase_email,
          supabase_password,
          supabase_authorization,
        );
        const data: dataElement[] = response.chunk_analysis;
        data.forEach((element) => {
          element.index = item.index;
          results.push(element);
        });

        // Queue a write operation - this ensures only one write happens at a time
        writeQueue.add(() => {
          fs.writeFileSync(output_file, JSON.stringify(results, null, 2), 'utf-8');
          console.log(`Updated output file with results from item ${item.index}`);
        });
      } catch (error) {
        console.error(`Error processing question: ${item.index}`, error);
      }
    }),
  );

  await Promise.all(tasks);
  await writeQueue.onIdle(); // Wait for all write operations to complete

  // fs.writeFileSync(output_file, JSON.stringify(results, null, 2), 'utf-8');
  console.info('All tasks completed.');
}

main().catch(console.error);
