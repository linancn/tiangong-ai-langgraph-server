import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import axios from 'axios';
import { z } from 'zod';

// OpenAI
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';

// Supabase
const base_url = process.env.BASE_URL ?? '';

type userElement = {
  grade?: string;
  major_background?: string;
  grasp_level?: string;
  major?: string;
  history?: string[];
};

type supabaseElement = {
  email: string;
  password: string;
  authorization: string;
};

const chainState = Annotation.Root({
  content: Annotation<string>(),
  supabase_auth: Annotation<supabaseElement>(),
  userData: Annotation<userElement>(),
  graphData: Annotation<string>(),
  refData: Annotation<string>(),
  portraitData: Annotation<string>(),
  pathData: Annotation<string[]>(),
});

async function getGraph(state: typeof chainState.State) {
  const url = process.env.NEO4J_URI ?? '';
  const username = process.env.NEO4J_USER ?? '';
  const password = process.env.NEO4J_PASSWORD ?? '';
  const graph = await Neo4jGraph.initialize({ url, username, password });
  const graphData = await graph.query(
    `
    CALL db.index.fulltext.queryNodes("concept_fulltext_index", "${state.content}") 
    YIELD node, score
    WITH node, score
    ORDER BY score DESC
    LIMIT 1
    OPTIONAL MATCH p1 = (:Concept)-[r1:HAS_PART]->(n)-[*1..5]-()
    WHERE n.id = node.id
    OPTIONAL MATCH p2 = (m)-[*1..3]-()-[]-()
    WHERE m.id = node.id
    RETURN COALESCE(p1, p2, '') AS result LIMIT 300
    `,
  );
  const graphData_json = JSON.stringify(graphData);
  const graphData_list: string[] = [];
  JSON.parse(graphData_json).map(
    (path: {
      result: Array<{
        CATEGORY?: string;
        id?: string;
      }>;
    }) => {
      const pathData_set = new Set<string>();
      path['result'].map((data) => {
        if (data.CATEGORY && data.id) {
          pathData_set.add(data.id);
        }
      });
      graphData_list.push(Array.from(pathData_set).join('>'));
    },
  );
  const graphData_string: string = graphData_list.join('\n');
  return { graphData: graphData_string };
}

async function getRefs(state: typeof chainState.State) {
  const url = `${base_url}/textbook_search`;
  const headers = {
    email: state.supabase_auth.email,
    password: state.supabase_auth.password,
    Authorization: `Bearer ${state.supabase_auth.authorization}`,
    'Content-Type': 'application/json',
  };

  const payload = {
    query: state.content,
    topK: 1,
    extK: 2,
  };
  const textbook_chunks = await axios.post(url, payload, { headers });
  const textbook_chunks_data: { source: string; content: string }[] = textbook_chunks.data;

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const response = await model.invoke([
    {
      role: 'assistant',
      content: `你是一位专业的教育内容分析助手。你的任务是分析教材内容，围绕用户提出的问题或主题"${state.content}"，识别并总结需要掌握的关键知识点，给出一段简洁明了的知识总结。`,
    },
    {
      role: 'human',
      content: `${textbook_chunks_data.map((item) => item.content || '').join('\n') || (textbook_chunks.data ?? '无')}`,
    },
  ]);

  return { refData: response.content };
}

async function getPortrait(state: typeof chainState.State) {
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const response = await model.invoke([
    {
      role: 'assistant',
      content: `你是一位专业的教育分析助手。你的任务是基于学生信息进行用户画像分析，概括学生的知识掌握情况以及学习关注点与需求（不需要给出建议），确保信息全面但简洁明了。`,
    },
    {
      role: 'human',
      content: `学生情况：
      -年级（用于判断学习阶段和相应需求）：${state.userData.grade}，
      -之前专业（仅适用于硕士/博士生，否则为空）：${state.userData.major ?? '无'}，
      -专业背景（用于评估其学科基础）：${state.userData.major_background}，
      -掌握程度（用于衡量知识熟练度）：${state.userData.grasp_level}，
      -过往提出的问题（用于分析其关注点和可能的知识盲区）：${state.userData.history?.join(', ') ?? '无'}。`,
    },
  ]);
  return { portraitData: response.content };
}

async function getKnowledge(state: typeof chainState.State) {
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const response = await model.invoke([
    {
      role: 'assistant',
      content: `你是一位智能学习顾问，负责根据学生提出的问题（${state.content ?? '无'}），请根据知识点总结，对初始知识体系进行优化。你需要识别并去除与问题关联性小、过于细化可以合并的冗余点以及与教材内容和事实相悖的知识点。最终给出一个合理的知识体系（知识点列表及其相互关系），用“->”表示先后关系。`,
    },
    {
      role: 'human',
      content: `以下是从Neo4j数据库提取的初始知识体系（初始知识点列表及其相互关系）：${state.graphData ?? '无'}。`,
    },
    {
      role: 'human',
      content: `以下是主要知识点总结：${state.refData ?? '无'}。`,
    },
  ]);
  console.log(response);
  return { graphData: response.content };
}

async function getPath(state: typeof chainState.State) {
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const pathResult = z.object({
    path: z.string().describe('知识点顺序学习列表，从初始知识点到最终知识点,用->分隔'),
  });

  const structuredLlm = model.withStructuredOutput(pathResult);

  const response = await structuredLlm.invoke([
    {
      role: 'assistant',
      content: `
      你是一位专业的学习路径规划专家，你的任务是基于学生当前水平，生成最简可行的学习路径。请严格按照以下要求进行分析和输出：
      1. 分析学习需求：结合学生的问题（${state.content ?? '无'}）及学习特点，明确需要掌握的关键知识点。 
      2. 识别关键知识点：结合提供的知识体系，提炼最核心的知识点（如形成过程、生物与化学性质、资源化利用相关要点、必要的基础化学与材料科学知识等）。 
      3. 构建线性学习路径： 
      - 严格遵循知识点的逻辑依赖关系（A→B 表示必须先掌握A，才能理解B）。 
      - 仅保留最核心的知识点，避免冗余，确保路径最短且最有效。 
      - 输出格式必须为严格的线性序列，即 A→B→C→D，不能出现并列项（如 A→B 且 A→C）。 
      - 知识点表述必须简洁明了，避免冗长的解释。
      4. 输出格式：最终输出的学习路径应为知识点顺序学习列表，从初始知识点到最终知识点，用“->”分隔,示例：基础化学知识→矿物化学（硅酸盐及其活性成分）→ 煤矸石的形成→煤矸石的化学组分→煤矸石的活性因素→资源化利用方法->矿物回收->建筑材料应用。`,
    },
    {
      role: 'human',
      content: `知识体系：${state.graphData ?? '无'}`,
    },
    {
      role: 'human',
      content: `学生学习特点与需求:${state.portraitData ?? '无'}。`,
    },
  ]);
  return { pathData: response['path'] };
}

const workflow = new StateGraph(chainState)
  .addNode('getGraph', getGraph)
  .addNode('getRefs', getRefs)
  .addNode('getPortrait', getPortrait)
  .addNode('getPath', getPath)
  .addNode('getKnowledge', getKnowledge)
  .addEdge('__start__', 'getGraph')
  .addEdge('__start__', 'getRefs')
  .addEdge('__start__', 'getPortrait')
  .addEdge('getGraph', 'getKnowledge')
  .addEdge('getRefs', 'getKnowledge')
  .addEdge('getPortrait', 'getKnowledge')
  .addEdge('getKnowledge', 'getPath')
  .addEdge('getPath', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
