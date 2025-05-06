import { ChatOpenAI } from '@langchain/openai';
import dotenv from 'dotenv';

dotenv.config();

const model = new ChatOpenAI({
  modelName: 'gpt-4.1',
  temperature: 0,
});

const schema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'https://example.com/product.schema.json',
  title: 'ResponseFormatter',
  type: 'object',
  properties: {
    answer: {
      description: "The answer to the user's question",
      type: 'string',
    },
    followup_question: {
      description: 'A followup question the user could ask',
      type: 'string',
    },
  },
  required: ['answer', 'followup_question'],
};

const modelWithStructure = model.withStructuredOutput(schema);

async function main() {
  const aiMsg = await modelWithStructure.invoke('What is the powerhouse of the cell?');
  console.log(aiMsg);
}

main().catch(console.error);
