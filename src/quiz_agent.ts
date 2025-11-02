import { Annotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

const openai_chat_model_mini = process.env.OPENAI_CHAT_MODEL_REASONNING_MINI ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';

type userElement = {
  grade?: string;
  major_background?: string;
  grasp_level?: string;
  major?: string;
  history?: string[];
  student_id?: string;
};

const singleChoiceSchema = z.object({
  Type: z.literal('SingleChoice'),
  TypeText: z.literal('单选题'),
  ProblemType: z.literal(1),
  Body: z.string().describe('题干内容（中文）。不要在此处包含选项标签或答案。'),
  Options: z
    .array(
      z.object({
        key: z.enum(['A', 'B', 'C', 'D']),
        value: z.string().describe('选项内容（中文）。'),
      }),
    )
    .min(2)
    .max(4)
    .superRefine((options, ctx) => {
      const allowedOrder = ['A', 'B', 'C', 'D'];
      const seenKeys = new Set<string>();
      options.forEach((option, index) => {
        if (seenKeys.has(option.key)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: '选项键不能重复。',
            path: [index, 'key'],
          });
        }
        seenKeys.add(option.key);
      });
      if (options.length > 0) {
        if (options[0].key !== 'A') {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: '第一项必须使用选项 A。',
            path: [0, 'key'],
          });
        }
        for (let i = 1; i < options.length; i += 1) {
          const expectedKey = allowedOrder[i];
          if (options[i].key !== expectedKey) {
            ctx.addIssue({
              code: z.ZodIssueCode.custom,
              message: '选项必须按照 A、B、C、D 的顺序连续排列，不得跳过。',
              path: [i, 'key'],
            });
          }
        }
      }
    })
    .describe('提供 2-4 个选项，键按从 A 开始连续排列。'),
  Answer: z
    .array(z.enum(['A', 'B', 'C', 'D']))
    .length(1)
    .describe('仅有 1 个正确答案，用其选项键表示。'),
  difficulty: z.number().describe('难度等级 1（易）至 5（难），与样题难度一致。'),
  Remark: z.string().describe('说明答案正确的原因，并引用相关知识内容。'),
});

const multipleChoicesSchema = z.object({
  Type: z.literal('MultipleChoice'),
  TypeText: z.literal('多选题'),
  ProblemType: z.literal(2),
  Body: z.string().describe('题干内容（中文）。不要包含选项标签或答案。'),
  Options: z
    .array(
      z.object({
        key: z.enum(['A', 'B', 'C', 'D']),
        value: z.string().describe('选项内容（中文）。'),
      }),
    )
    .length(4)
    .describe('提供 A-D 共 4 个选项。'),
  Answer: z
    .array(z.enum(['A', 'B', 'C', 'D']))
    .min(2)
    .describe('至少 2 个正确答案，用其选项键表示。'),
  difficulty: z.number().describe('难度等级 1（易）至 5（难），与样题难度一致。'),
  Remark: z.string().describe('解释每个选项正确或错误的原因。'),
});

const shortAnswerSchema = z.object({
  Type: z.literal('ShortAnswer'),
  TypeText: z.literal('主观题'),
  ProblemType: z.literal(5),
  Body: z.string().describe('题干内容（中文），引导学习者进行开放式作答。'),
  difficulty: z.number().describe('难度等级 1（易）至 5（难），与样题难度一致。'),
  Remark: z.string().describe('参考知识内容给出示例答案或评分要点。'),
});

const responseSchema = z.object({
  questions: z
    .array(z.union([singleChoiceSchema, multipleChoicesSchema, shortAnswerSchema]))
    .length(3)
    .describe('严格输出 3 道题。每题应根据对应样题选择匹配的题型架构。'),
});

type GeneratedQuestion = z.infer<typeof responseSchema>['questions'][number];

const chainState = Annotation.Root({
  knowledge_point_name: Annotation<string>(),
  knowledge_content: Annotation<string>(),
  sample_questions: Annotation<string[]>(),
  user_context: Annotation<userElement>(),
  output: Annotation<GeneratedQuestion[]>(),
});

async function GenerateQuizQuestions(state: typeof chainState.State) {
  const userContext = state.user_context ?? {};
  const today = new Date().toISOString().slice(0, 10);
  const seedComponents = [
    userContext.student_id ?? '',
    userContext.grade ?? '',
    userContext.major ?? '',
    today,
  ].filter((component) => component && component.length > 0);
  const variantSeed = seedComponents.length > 0 ? seedComponents.join('|') : today;

  const userContextSummary =
    Object.keys(userContext).length === 0
      ? '无'
      : [
          userContext.student_id ? `学生ID: ${userContext.student_id}` : '',
          userContext.grade ? `年级: ${userContext.grade}` : '',
          userContext.major ? `专业: ${userContext.major}` : '',
          userContext.major_background ? `专业背景: ${userContext.major_background}` : '',
          userContext.grasp_level ? `掌握程度: ${userContext.grasp_level}` : '',
          (userContext.history?.length ?? 0) > 0
            ? `历史错题: ${userContext.history?.join('； ')}`
            : '',
        ]
          .filter((entry) => entry.length > 0)
          .join('； ');

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    model: openai_chat_model_mini,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(responseSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content: `你是一名教学设计专家，负责根据给定的样题生成高度相似且不重复的练习题。
请严格遵循以下要求：
- 语言：全程使用中文编写题干、选项、答案与解析。
- 题型规范：每道题必须严格匹配以下之一的结构：
  * 单选题：仅 1 个正确选项。
  * 多选题：存在多个正确选项。
  * 主观题：开放式作答，无选项。
- 样题映射：逐条分析样题，并一一对应生成新题：
  * 总共输出 3 道题，顺序与输入样题一致。
  * 若样题为单选题或问答且存在唯一答案，则输出单选题。
  * 若样题为多选题或需要多个关键要点，则输出多选题。
  * 若样题要求开放式回答，则输出主观题。
  * 若样题为判断题（对/错），请转换为单选题，且仅提供两个选项：A：正确、B：错误，不得出现其他选项，并设置唯一正确答案。
- 知识对齐：题干、答案与解析必须与提供的知识内容一致，并明确指向给定的知识点。
- 相似不重复：保持与样题相同的题型与思维路径，但更换情境、角色、数据或措辞，避免与样题重复。
- 结构映射：保持与样题相同的逻辑结构、难度与认知要求，同时更换表述与背景以避免重复。
- 多样性：三道题应体现不同角度或应用场景。
- 难度：每题的难度值（1-5）应与其对应样题的隐含难度保持一致。
- 解析：请在解析中明确将推理或关键点与知识内容对应。
- 变体种子（用于不同学生有差异）：${variantSeed}。
- 学生上下文：如有可用，在不改变题型的前提下，将其背景、掌握程度或历史错题细节融合到题面中。`,
    },
    {
      role: 'human',
      content: `知识点名称: ${state.knowledge_point_name ?? ''}\n知识点内容: ${
        state.knowledge_content ?? ''
      }\n请基于该知识点生成三道新题目。`,
    },
    {
      role: 'human',
      content: `学生上下文: ${userContextSummary}`,
    },
    {
      role: 'human',
      content: `示例题目（逐条对应生成一题）：\n${(state.sample_questions ?? [])
        .map((question, index) => `${index + 1}. ${question}`)
        .join('\n')}`,
    },
  ]);

  return { output: response.questions };
}

const workflow = new StateGraph(chainState)
  .addNode('GenerateQuizQuestions', GenerateQuizQuestions)
  .addEdge('__start__', 'GenerateQuizQuestions')
  .addEdge('GenerateQuizQuestions', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
