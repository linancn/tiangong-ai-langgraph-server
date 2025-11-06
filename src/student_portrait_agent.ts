import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import axios from 'axios';
import FormData from 'form-data';
import { basename } from 'node:path';
import { z } from 'zod';

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model =
  process.env.OPENAI_CHAT_MODEL_REASONNING ?? process.env.OPENAI_CHAT_MODEL ?? '';
const mineru_base_url = process.env.MINERU_BASE_URL ?? '';
const mineru_api_key = process.env.MINERU_API_KEY ?? '';
const mineru_vision_provider = process.env.MINERU_VISION_PROVIDER ?? '';
const mineru_vision_model = process.env.MINERU_VISION_MODEL ?? '';
const mineru_vision_prompt = process.env.MINERU_VISION_PROMPT ?? '';
const mineru_chunk_type_flag = process.env.MINERU_CHUNK_TYPE ?? 'true';
const mineru_pretty_flag = process.env.MINERU_PRETTY ?? 'true';
const student_portrait_bucket = process.env.STUDENT_PORTRAIT_BUCKET ?? '';
const student_portrait_bucket_region = process.env.STUDENT_PORTRAIT_BUCKET_REGION ?? '';
const student_portrait_bucket_access_key = process.env.STUDENT_PORTRAIT_BUCKET_ACCESS_KEY ?? '';
const student_portrait_bucket_secret_key = process.env.STUDENT_PORTRAIT_BUCKET_SECRET_KEY ?? '';

let cachedS3Client: S3Client | undefined;

function getS3Client() {
  if (!cachedS3Client) {
    const hasStaticCredentials =
      student_portrait_bucket_access_key.length > 0 &&
      student_portrait_bucket_secret_key.length > 0;

    cachedS3Client = new S3Client({
      region: student_portrait_bucket_region || undefined,
      credentials: hasStaticCredentials
        ? {
            accessKeyId: student_portrait_bucket_access_key,
            secretAccessKey: student_portrait_bucket_secret_key,
          }
        : undefined,
    });
  }
  return cachedS3Client;
}

// gradeEnum 与 scoreSchema
// - 用途: 这些 schema 用于约束模型的结构化输出（即 AI 生成的评分内容应满足这些约束）。
// - 来源: 由本程序定义，模型输出需遵循（structured LLM 输出时验证）。
// - 注意: scoreSchema 描述的实例（如 process_assessment.score、performance_assessment.overall_score）为 AI 生成结果。
const gradeEnum = z.enum(['A', 'B', 'C', 'D', 'E']);

const scoreSchema = z.object({
  // AI 生成: 0-100 的数值型总分
  total_score: z.number().min(0).max(100).describe('0-100之间的综合得分'),
  // AI 生成: 基于 total_score 的等级（A-E）
  grade: gradeEnum.describe('根据总分映射的等级 A-E'),
  // AI 生成: 对得分的简要解释（必须基于输入证据）
  interpretation: z.string().describe('简要说明评分依据、整体表现'),
  // AI 生成: 用于支撑判定的证据片段列表（优先引用输入数据）
  // evidence: z.array(z.string()).describe('支撑该评分的关键证据').default([]),
});

// 本地截断/摘要长度常量（运行时内部使用）
// 说明：保护 LLM 上下文长度，避免因输入过长导致请求失败或被中途取消（aborted）。
const TEXT_SUMMARY_MAX_LENGTH = 50000; // 针对传入 LLM 的 JSON/文本做保护性截断

// StudentInfo: 学生基础信息结构
// 来源: 外部输入（由调用方/上层服务提供）。字段均为可选，具体使用场景可酌情要求必填项。
type StudentInfo = {
  grade?: string;
  major_background?: string;
  grasp_level?: string;
  major?: string;
  user_id?: number;
  // additional_notes?: string;
};

async function uploadPortraitMarkdownToS3(markdown: string, studentInfo?: StudentInfo) {
  const userId = studentInfo?.user_id;
  if (!userId) {
    return;
  }

  const bucket = student_portrait_bucket;
  const objectKey = String(userId);

  const client = getS3Client();
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: objectKey,
    Body: Buffer.from(markdown, 'utf-8'),
    ContentType: 'text/markdown; charset=utf-8',
  });

  try {
    await client.send(command);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Failed to upload student portrait markdown to S3 for user ${objectKey}: ${message}`,
    );
  }
}

// AnswerDetail: 单题详情（来源通常为学习计划或练习数据，属于外部输入）
type AnswerDetail = {
  stem?: string;
  opts?: Array<string | number>;
  ans?: Array<string | number>;
  user_ans?: Array<string | number>;
  score?: number | string;
  analysis?: string;
  feedback?: string;
  knowledge_points?: string[];
};

// PlanDetailNode: 学习计划/知识节点（外部输入），可递归包含子节点与答题记录
type PlanDetailNode = {
  node_id?: string;
  node_name?: string;
  conversations?: string[];
  answer_details?: AnswerDetail[];
  plan_detail?: PlanDetailNode[];
};

// ArtifactAnalysis: 对作品的解析结果，主要由外部解析服务（mineru）或本地处理生成
// - mineru_payload_* 字段为文档拆解服务返回的元数据（外部服务产物）。
// - summary 一般为文档拆解服务或后续模型整理出的摘要（AI/服务生成）。
// - errors 列出调用或解析过程中的异常信息（本地代码填充）。
type ArtifactAnalysis = {
  file_url: string;
  // mineru_payload_excerpt?: string;
  mineru_payload_length?: number;
  // mineru_payload_truncated?: boolean;
  summary: string;
  errors?: string[];
};
// portraitSchema: 最终学生画像的结构定义（AI 生成的输出应符合此 schema）
// 说明：portraitSchema 中的 process_assessment / performance_assessment 等均由模型（AI）生成并应满足结构化约束。
const portraitSchema = z.object({
  overview: z
    .string()
    .describe('对学生在该主题下整体学习状态的高度概括，点出过程与最终表现的核心结论。'),
  process_assessment: z.object({
    score: scoreSchema.describe('过程考查的整体评分结果，需结合学习过程表现给出0-100分与等级'),
    knowledge_structure: z
      .array(
        z.object({
          node_id: z.string().nullable(),
          node_name: z.string().describe('知识节点或子主题名称'),
          mastery_level: z.string().describe('该知识点的掌握度评价，可包含等级与简短说明。'),
          strengths: z.array(z.string()).describe('该知识点表现良好的方面').default([]),
          issues: z.array(z.string()).describe('存在的问题、错误类型或理解薄弱点').default([]),
          recommended_actions: z
            .array(z.string())
            .describe('针对该知识点的改进建议或练习方向')
            .default([]),
          // evidence: z
          //   .array(z.string())
          //   .describe('引用的客观证据，如题目表现、提问记录等')
          //   .default([]),
        }),
      )
      .describe('针对知识体系中每个节点的掌握情况分析')
      .default([]),
    metrics: z.object({
      mastery: z.object({
        level: z.string().describe('综合掌握度评价结果，需明确等级并说明其含义'),
        interpretation: z.string().describe('对掌握度评价的解释与背后原因'),
        // evidence: z.array(z.string()).describe('支撑掌握度判断的关键证据').default([]),
      }),
      stability: z.object({
        level: z.string().describe('稳定性水平，如稳定、波动较大等，需要有定性或定量描述'),
        interpretation: z.string().describe('稳定性表现的解析与潜在原因'),
        // evidence: z
        //   .array(z.string())
        //   .describe('支撑稳定性判断的关键证据，例如多次作答的正确率变化、提问波动等')
        //   .default([]),
        fluctuations: z.array(z.string()).describe('识别到的波动或不稳定环节').default([]),
      }),
      transfer: z.object({
        level: z.string().describe('迁移度水平，如较强/一般/较弱等，需要明确评价标准'),
        interpretation: z.string().describe('对迁移能力的分析，包括在综合或跨知识点题目中的表现'),
        // evidence: z.array(z.string()).describe('支撑迁移度判断的案例或表现').default([]),
        gaps: z.array(z.string()).describe('迁移过程中暴露出的不足或卡点').default([]),
      }),
    }),
    next_steps: z.object({
      priorities: z.array(z.string()).describe('下一阶段最重要的关注重点，按优先级排序'),
      recommended_practices: z.array(z.string()).describe('具体的练习或学习任务建议').default([]),
      monitoring_indicators: z.array(z.string()).describe('建议持续追踪的指标或检查点').default([]),
    }),
    communication_notes: z
      .array(z.string())
      .describe('与学生或家长沟通时的注意事项、鼓励点或风险提示')
      .default([]),
  }),
  performance_assessment: z.object({
    overall_score: scoreSchema.describe('最终表现的综合评分结果，需给出0-100分与等级'),
    overall_summary: z
      .string()
      .describe('对最终作品整体表现性评价的综合总结，需突出关键结论与等级判断'),
    artifacts: z
      .array(
        z.object({
          file_url: z.string().describe('作品文件链接'),
          ratings: z.object({
            content_quality: z.object({
              score: z.number().min(0).max(100),
              level: z.string(),
              comments: z.string(),
            }),
            thinking_innovation: z.object({
              score: z.number().min(0).max(100),
              level: z.string(),
              comments: z.string(),
            }),
            expression_presentation: z.object({
              score: z.number().min(0).max(100),
              level: z.string(),
              comments: z.string(),
            }),
            norms_reflection: z.object({
              score: z.number().min(0).max(100),
              level: z.string(),
              comments: z.string(),
            }),
          }),
          total_score: z.number().min(0).max(100),
          grade: gradeEnum,
          strengths: z.array(z.string()).default([]),
          issues: z.array(z.string()).default([]),
          improvement_actions: z.array(z.string()).default([]),
          // evidence: z.array(z.string()).default([]),
        }),
      )
      .default([]),
    monitoring_recommendations: z.array(z.string()).default([]),
  }),
});

// 衍生 schema：用于对不同阶段/环节的 AI 输出进行结构化约束
// - processAssessmentSchema / performanceAssessmentSchema: 用于约束对应的 AI 输出（由模型生成）
const processAssessmentSchema = portraitSchema.shape.process_assessment;
const performanceAssessmentSchema = portraitSchema.shape.performance_assessment;
// portraitOverviewSchema: 用于生成简短概述（AI 生成）
const portraitOverviewSchema = z.object({
  overview: z.string().describe('综合过程考查与最终表现后对学生水平与改进重点的 2-3 句总结。'),
});
const portraitMarkdownSchema = z.object({
  markdown: z
    .string()
    .describe(
      '最终呈现给教师或家长的 Markdown 画像，应包含过程考查、最终表现、下一步建议等结构化部分。',
    ),
});

// Type aliases: 便于在代码中引用这些结构化类型
// 注意：这些类型大部分对应的是 AI/模型生成的内容（由 structured LLM 输出并用这些 schema 验证）。
type StudentPortrait = z.infer<typeof portraitSchema>;
type ProcessAssessment = z.infer<typeof processAssessmentSchema>;
type PerformanceAssessment = z.infer<typeof performanceAssessmentSchema>;

// chainState: 状态图（StateGraph）的根注解定义
// 说明每个字段的来源（外部输入 / 本地生成 / AI 生成）及必填性：
// - 输入（外部）: theme、student_info、plan_detail、final_artifacts 等由外部提交的数据
// - 本地生成: plan_summary（由 summarizePlanDetail 生成）、artifact_summary（由 buildArtifactSummary 生成）、portrait_json 等
// - AI（模型）生成: process_assessment、performance_assessment、portrait、portrait_markdown（由相应的 assess*/generate/render 节点生成）
const chainState = Annotation.Root({
  // 学习主题/课程（外部输入，建议提供以提高提示质量）
  theme: Annotation<string>(),
  // 学生基础信息（外部输入，可选）
  student_info: Annotation<StudentInfo>(),
  // 学习计划与答题记录（外部输入，可选）
  plan_detail: Annotation<PlanDetailNode[]>(),
  // 学习计划摘要（本地/工作流生成，由 summarizePlanDetail 填充）
  plan_summary: Annotation<string>(),
  // 最终作品文件链接列表（外部输入），每个元素应为可访问的 file_url
  final_artifacts: Annotation<string[]>(),
  // 过程考查（AI 生成，assessProcess 产出，结构受 processAssessmentSchema 约束）
  process_assessment: Annotation<ProcessAssessment>(),
  // 最终表现评分（AI 生成，assessPerformance 产出，结构受 performanceAssessmentSchema 约束）
  performance_assessment: Annotation<PerformanceAssessment>(),
  // 作品解析（由 analyzeArtifacts 调用外部文档拆解服务或本地处理生成；可累加）
  artifact_analysis: Annotation<ArtifactAnalysis[]>({
    reducer: (x, y) => {
      const base = Array.isArray(x) ? x : [];
      if (!y) {
        return base;
      }
      return base.concat(y);
    },
    default: () => [],
  }),
  // 作品解析汇总（本地生成，基于 artifact_analysis）
  artifact_summary: Annotation<string>(),
  // 最终学生画像结构（由 generatePortrait/模型生成）
  portrait: Annotation<StudentPortrait>(),
  // portrait 的 JSON 字符串（本地生成，便于存储或传输）
  portrait_json: Annotation<string>(),
  // 最终 Markdown 输出（由 renderMarkdown 生成，AI 帮助排版，但内容基于 portrait）
  portrait_markdown: Annotation<string>(),
});

function normalizeValue(value: string | number | null | undefined): string {
  if (value === null || value === undefined) {
    return '';
  }
  return value.toString().trim();
}

function truncateText(value: string, maxLength = TEXT_SUMMARY_MAX_LENGTH): string {
  if (!value) {
    return '';
  }
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength)}\n...[已截断，原始长度 ${value.length} 字符]`;
}

function computeLocalStats(node?: PlanDetailNode) {
  let total = 0;
  let answered = 0;
  let correct = 0;

  node?.answer_details?.forEach((detail) => {
    const standard = (detail.ans ?? []).map(normalizeValue).filter(Boolean);
    const user = (detail.user_ans ?? []).map(normalizeValue).filter(Boolean);
    if (standard.length > 0) {
      total += 1;
    }
    if (user.length > 0) {
      answered += 1;
    }
    if (standard.length > 0 && user.length > 0) {
      const standardSet = new Set(standard);
      const userSet = new Set(user);
      const isExactMatch =
        standardSet.size === userSet.size &&
        Array.from(standardSet).every((entry) => userSet.has(entry));
      if (isExactMatch) {
        correct += 1;
      }
    }
  });

  return { total, answered, correct };
}

function aggregateStats(nodes?: PlanDetailNode[]): {
  total: number;
  answered: number;
  correct: number;
} {
  return (nodes ?? []).reduce(
    (acc, node) => {
      const local = computeLocalStats(node);
      acc.total += local.total;
      acc.answered += local.answered;
      acc.correct += local.correct;
      const child = aggregateStats(node.plan_detail);
      acc.total += child.total;
      acc.answered += child.answered;
      acc.correct += child.correct;
      return acc;
    },
    { total: 0, answered: 0, correct: 0 },
  );
}

function summarizePlanNodes(
  nodes: PlanDetailNode[] | undefined,
  depth = 0,
  path: string[] = [],
): string[] {
  if (!nodes || nodes.length === 0) {
    return [];
  }

  return nodes.flatMap((node) => {
    const indent = '  '.repeat(depth);
    const displayName = node.node_name ?? `未命名节点${node.node_id ?? ''}`;
    const currentPath = [...path, displayName].join(' > ');
    const lines: string[] = [];

    lines.push(`${indent}- 节点: ${displayName}${node.node_id ? ` (ID: ${node.node_id})` : ''}`);
    lines.push(`${indent}  路径: ${currentPath}`);

    const localStats = computeLocalStats(node);
    if (localStats.total > 0 || localStats.answered > 0) {
      const accuracy =
        localStats.total > 0
          ? `${((localStats.correct / localStats.total) * 100).toFixed(1)}%`
          : '无可比对';
      lines.push(
        `${indent}  答题统计: 标准题目${localStats.total}题，已作答${localStats.answered}题，参考答案完全匹配${localStats.correct}题，正确率${accuracy}`,
      );
    }

    if (node.conversations?.length) {
      lines.push(`${indent}  提问记录: ${node.conversations.join('； ')}`);
    }

    node.answer_details?.forEach((detail, index) => {
      lines.push(`${indent}  练习${index + 1}: ${detail.stem ?? '未提供题干'}`);
      if (detail.opts?.length) {
        lines.push(`${indent}    选项: ${detail.opts.map(normalizeValue).join(' | ')}`);
      }
      if (detail.ans?.length) {
        lines.push(`${indent}    参考答案: ${detail.ans.map(normalizeValue).join(' | ')}`);
      }
      if (detail.user_ans?.length) {
        lines.push(`${indent}    学生作答: ${detail.user_ans.map(normalizeValue).join(' | ')}`);
      }
      if (detail.score !== undefined && detail.score !== null && detail.score !== '') {
        lines.push(`${indent}    得分或评价: ${detail.score}`);
      }
      if (detail.analysis) {
        lines.push(`${indent}    解析: ${detail.analysis}`);
      }
      if (detail.feedback) {
        lines.push(`${indent}    反馈: ${detail.feedback}`);
      }
      if (detail.knowledge_points?.length) {
        lines.push(`${indent}    涉及知识点: ${detail.knowledge_points.join(' | ')}`);
      }
    });

    if (node.plan_detail?.length) {
      lines.push(...summarizePlanNodes(node.plan_detail, depth + 1, [...path, displayName]));
    }

    return lines;
  });
}

function buildPlanSummary(nodes?: PlanDetailNode[]): string {
  if (!nodes || nodes.length === 0) {
    return '未提供学习计划与作答数据。';
  }

  const totals = aggregateStats(nodes);
  const accuracy =
    totals.total > 0 ? `${((totals.correct / totals.total) * 100).toFixed(1)}%` : '无可比对';

  const header = `整体答题统计: 标准题目${totals.total}题，已作答${totals.answered}题，参考答案完全匹配${totals.correct}题，正确率${accuracy}`;
  const detailLines = summarizePlanNodes(nodes);
  return [header, ...detailLines].join('\n');
}

function formatStudentInfo(info?: StudentInfo): string {
  if (!info) {
    return '未提供学生信息。';
  }

  const parts = [
    info.grade ? `年级: ${info.grade}` : null,
    info.major ? `专业: ${info.major}` : null,
    info.major_background ? `专业背景: ${info.major_background}` : null,
    info.grasp_level ? `自评掌握程度: ${info.grasp_level}` : null,
    // info.additional_notes ? `补充信息: ${info.additional_notes}` : null,
  ].filter(Boolean);

  return parts.length > 0 ? parts.join('； ') : '未提供详细的学生背景。';
}

async function summarizePlanDetail(state: typeof chainState.State) {
  const plan_summary = buildPlanSummary(state.plan_detail);
  return { plan_summary };
}

// function serializeMineruPayload(
//   payload: unknown,
//   // maxLength = MINERU_PAYLOAD_EXCERPT_LIMIT,
// ): { excerpt: string; totalLength: number; truncated: boolean } {
//   if (payload === null || payload === undefined) {
//     return { excerpt: '', totalLength: 0, truncated: false };
//   }

//   let text: string;
//   if (typeof payload === 'string') {
//     text = payload;
//   } else {
//     try {
//       text = JSON.stringify(payload, null, 2);
//     } catch {
//       text = String(payload);
//     }
//   }

//   // const truncated = text.length > maxLength;
//   // const excerpt = truncated ? truncateText(text, maxLength) : text;
//   return { excerpt, totalLength: text.length, truncated };
//   return { excerpt, totalLength: text.length, truncated };
// }

type MineruTextChunk = {
  text?: string;
  page_number?: number;
};

type MineruWithImagesResponse = {
  result?: MineruTextChunk[];
};

function extractFilenameFromUrl(fileUrl: string): string {
  try {
    const parsed = new URL(fileUrl);
    if (parsed.pathname) {
      const decoded = decodeURIComponent(parsed.pathname);
      const name = basename(decoded);
      if (name) {
        return name;
      }
    }
  } catch {
    // Ignore malformed URL and fall back to default name
  }
  return 'artifact';
}

type DownloadedArtifact = {
  buffer: Buffer;
  filename: string;
  contentType: string;
};

async function downloadArtifactFile(fileUrl: string): Promise<DownloadedArtifact> {
  const response = await axios.get<ArrayBuffer>(fileUrl, {
    responseType: 'arraybuffer',
    timeout: 120000,
  });

  const buffer = Buffer.from(response.data);
  const filename = extractFilenameFromUrl(fileUrl);
  const headerValue = response.headers?.['content-type'];
  const contentType =
    typeof headerValue === 'string' && headerValue.trim().length > 0
      ? headerValue.split(';')[0].trim()
      : 'application/octet-stream';

  return {
    buffer,
    filename,
    contentType,
  };
}

async function callMineruWithImages(file: DownloadedArtifact): Promise<MineruWithImagesResponse> {
  if (!mineru_base_url) {
    throw new Error('文档拆解服务未配置，请设置 MINERU_BASE_URL');
  }

  const form = new FormData();
  form.append('file', file.buffer, {
    filename: file.filename,
    contentType: file.contentType,
  });
  if (mineru_vision_prompt.trim().length > 0) {
    form.append('prompt', mineru_vision_prompt);
  }
  if (mineru_vision_provider.trim().length > 0) {
    form.append('provider', mineru_vision_provider);
  }
  if (mineru_vision_model.trim().length > 0) {
    form.append('model', mineru_vision_model);
  }

  const headers: Record<string, string> = {
    ...form.getHeaders(),
    Accept: 'application/json',
  };

  if (mineru_api_key) {
    headers.Authorization = `Bearer ${mineru_api_key}`;
  }

  const response = await axios.post<MineruWithImagesResponse>(mineru_base_url, form, {
    headers,
    timeout: 300000,
    maxBodyLength: Infinity,
    params: {
      chunk_type: mineru_chunk_type_flag ? 'true' : 'false',
      pretty: mineru_pretty_flag ? 'true' : 'false',
    },
  });

  return response.data;
}

function buildMineruSummary(payload: MineruWithImagesResponse): string {
  const chunks = Array.isArray(payload?.result) ? payload.result : [];
  if (chunks.length === 0) {
    return '';
  }

  const lines = chunks
    .map((chunk) => {
      const text = typeof chunk?.text === 'string' ? chunk.text.trim() : '';
      if (!text) {
        return '';
      }
      const pageNumberRaw = chunk?.page_number;
      const pageNumber =
        typeof pageNumberRaw === 'number'
          ? pageNumberRaw
          : Number.isFinite(Number(pageNumberRaw))
            ? Number(pageNumberRaw)
            : undefined;
      const pageLabel = pageNumber && pageNumber > 0 ? `第${pageNumber}页` : '未注明页码';
      return `[${pageLabel}] ${text}`;
    })
    .filter((line) => line.length > 0);

  return lines.length > 0 ? lines.join('\n\n') : '';
}

function buildArtifactSummary(analyses: ArtifactAnalysis[]): string {
  if (analyses.length === 0) {
    return '未提供最终作品或暂未完成解析。';
  }

  return analyses
    .map((analysis) => {
      const errorLine =
        analysis.errors && analysis.errors.length > 0
          ? `解析异常: ${analysis.errors.join('； ')}`
          : '';
      // const payloadLine =
      //   analysis.mineru_payload_excerpt && analysis.mineru_payload_excerpt.trim().length > 0
      //     ? `原始解析片段${analysis.mineru_payload_truncated ? '（已截断）' : ''}${
      //         analysis.mineru_payload_length
      //           ? `，原始长度约 ${analysis.mineru_payload_length} 字符`
      //           : ''
      //       }:\n${analysis.mineru_payload_excerpt}`
      //     : '';

      return [
        `解析摘要:\n${analysis.summary || '未获取到有效内容。'}`,
        errorLine,
        // payloadLine,
      ]
        .filter(Boolean)
        .join('\n');
    })
    .join('\n\n');
}

async function analyzeArtifacts(state: typeof chainState.State) {
  const artifacts = state.final_artifacts ?? [];

  if (artifacts.length === 0) {
    return {
      artifact_analysis: [],
      artifact_summary: '未提供最终作品或提交内容。',
    };
  }

  const analyses: ArtifactAnalysis[] = [];

  for (const artifactEntry of artifacts) {
    const fileUrl = typeof artifactEntry === 'string' ? artifactEntry.trim() : '';

    if (!fileUrl) {
      analyses.push({
        file_url: fileUrl || '未提供链接',
        summary: '未提供可供解析的文件链接。',
        errors: ['缺少 file_url'],
      });
      continue;
    }

    if (!mineru_base_url) {
      analyses.push({
        file_url: fileUrl,
        summary: '文档拆解服务未配置，无法解析作品内容。',
        errors: ['缺少 MINERU_BASE_URL 配置'],
      });
      continue;
    }

    const errors: string[] = [];
    // let mineru_payload_excerpt = '';
    let mineru_payload_length = 0;
    // let mineru_payload_truncated = false;
    let summary = '';

    try {
      const downloadedFile = await downloadArtifactFile(fileUrl);
      const mineruRawPayload = await callMineruWithImages(downloadedFile);
      // const payloadInfo = serializeMineruPayload(mineruRawPayload);
      // mineru_payload_excerpt = payloadInfo.excerpt;
      // mineru_payload_length = payloadInfo.totalLength;
      // mineru_payload_truncated = payloadInfo.truncated;
      summary = buildMineruSummary(mineruRawPayload);
      // if (!summary) {
      //   summary = payloadInfo.excerpt;
      // }
    } catch (error) {
      errors.push(
        error instanceof Error
          ? `调用文档拆解服务失败: ${error.message}`
          : '调用文档拆解服务失败且未提供错误信息。',
      );
    }

    if (!summary) {
      summary = '文档拆解服务未返回摘要，可手动检查原始内容。';
    }
    // summary = truncateText(summary);

    analyses.push({
      file_url: fileUrl,
      // mineru_payload_excerpt: mineru_payload_excerpt || undefined,
      mineru_payload_length: mineru_payload_length || undefined,
      // mineru_payload_truncated: mineru_payload_truncated || undefined,
      summary,
      errors: errors.length > 0 ? errors : undefined,
    });
  }

  const artifact_summary = buildArtifactSummary(analyses);
  return {
    artifact_analysis: analyses,
    artifact_summary,
  };
}

async function assessProcess(state: typeof chainState.State) {
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(processAssessmentSchema);
  // const planDetailJson = JSON.stringify(state.plan_detail ?? [], null, 2);

  const process_assessment = await structuredLlm.invoke([
    {
      role: 'system',
      content: `你是一位具备教育测评与学习科学背景的教学诊断专家。请根据提供的学习计划、答题表现与互动记录，输出过程考查（process_assessment）结论，并严格遵循给定的 JSON schema。使用时先阅读摘要把握全局指标，再结合 JSON 证据补充细节。需要：
1. 针对知识体系中每个节点梳理掌握度、优势、问题、改进行动与证据；
2. 对掌握度、稳定性、迁移度三大维度给出等级判定、解释及支撑证据（如有波动或迁移不足需列出）；
3. 明确下一步的重点、练习建议与监测指标；
4. 给出沟通要点或风险提示；
5. 结合学习投入、完成度、互动与反思等证据，按照提供的评分等级表计算过程考查的 total_score（0-100）与 grade（A-E），并说明评分依据与关键证据。
请只基于提供的数据，不得臆造。`,
    },
    {
      role: 'human',
      content: `学习主题或课程: ${state.theme || '未提供'}`,
    },
    {
      role: 'human',
      content: `学生基础信息:\n${formatStudentInfo(state.student_info)}`,
    },
    {
      role: 'human',
      content: `学习计划与作答摘要（仅呈现全局指标与趋势，供你把握整体水平）:\n${state.plan_summary || '无'}`,
    },
    {
      role: 'human',
      content: `过程考查评分等级说明:
| 等级 | 分数区间 | 特征描述 |
| --- | --- | --- |
| A | 90-100 | 全程积极投入，学习频率高；能自主查找资料，知识掌握稳步提升；与同伴交流频繁，反思深刻、具有独立见解。 |
| B | 80-89 | 学习态度认真，完成任务及时；掌握水平有明显提升；能主动参与讨论并做出反思。 |
| C | 70-79 | 学习投入较稳定，能按要求完成任务；有一定提升但不显著；反思简单、合作有限。 |
| D | 60-69 | 学习积极性不足；任务常延迟或流于形式；进步缓慢或波动大；反思浅显。 |
| E | <60 | 学习参与度低，任务缺失严重；无明显成长；缺乏合作与反思。 |
请据此确定 total_score 与 grade，并在 interpretation 中写清判定理由。`,
    },
  ]);

  return { process_assessment };
}

async function assessPerformance(state: typeof chainState.State) {
  // 允许通过环境变量控制 OpenAI 请求超时时间（毫秒），避免长时间阻塞导致的 AbortError
  const openaiTimeoutMs = Number(process.env.OPENAI_TIMEOUT_MS ?? '0');

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
    // 保护性超时（仅当 >0 时生效）
    timeout: openaiTimeoutMs > 0 ? openaiTimeoutMs : undefined,
  });

  const structuredLlm = model.withStructuredOutput(performanceAssessmentSchema);
  const artifactAnalysisJsonRaw = JSON.stringify(state.artifact_analysis ?? [], null, 2);
  const artifactAnalysisJson = truncateText(artifactAnalysisJsonRaw, TEXT_SUMMARY_MAX_LENGTH);
  const artifactSummarySafe = truncateText(state.artifact_summary || '', TEXT_SUMMARY_MAX_LENGTH);

  try {
    const performance_assessment = await structuredLlm.invoke([
      {
        role: 'system',
        content: `你是一位擅长表现性评价的教学测评专家。请基于提供的作品解析数据，输出最终表现（performance_assessment）结论，并严格遵循给定的 JSON schema。先参考摘要掌握整体评分走向，再用 JSON 证据支撑细节描述。
要求：
1. 对每件作品在“内容质量、思维与创新、表达与呈现、规范与反思”四个维度分别给出 0-100 分的量化评分、等级（A-E）与评语；
2. 结合证据列出作品亮点、问题、改进行动与支撑证据；
3. 给出整体总结和后续监测建议；
4. 严格依据数据，不得编造；
5. 依据评分等级表计算最终表现的 overall_score（total_score、grade、interpretation），明确综合得分与判定理由。
评分需参考以下等级说明：
| 等级 | 分数区间 | 特征描述 |
| --- | --- | --- |
| A | 90-100 | 逻辑严密、内容深入、创新显著、表达专业，具研究价值 |
| B | 80-89 | 内容扎实、思路清晰、略有创新、表达规范 |
| C | 70-79 | 内容基本正确、表达一般、创新性不足 |
| D | 60-69 | 存在明显逻辑或内容缺陷，表达欠清晰 |
| E | <60 | 内容错误多、结构混乱或可疑抄袭 |`,
      },
      {
        role: 'human',
        content: `学习主题或课程: ${state.theme || '未提供'}`,
      },
      {
        role: 'human',
        content: `学生基础信息:\n${formatStudentInfo(state.student_info)}`,
      },
      {
        role: 'human',
        content: `最终作品解析摘要（全局评分与总体结论，帮助你抓住宏观表现）:\n${artifactSummarySafe || '无'}`,
      },
    ]);

    return { performance_assessment };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // 常见：AbortError 或网络层抛出的 "aborted" 字样，通常由超时、连接中断或请求过大引起
    if (/aborted/i.test(message) || (error as any)?.name === 'AbortError') {
      throw new Error(
        `assessPerformance 被中断（aborted）。常见原因：\n- OpenAI 请求超时或被取消（可设置环境变量 OPENAI_TIMEOUT_MS 增大超时）；\n- 上下文过长导致请求过大（已做保护性截断，仍建议减少作品解析长度/数量）；\n- 短时网络中断或对端关闭连接。\n原始错误：${message}`,
      );
    }
    throw new Error(`assessPerformance 调用失败：${message}`);
  }
}

async function generatePortrait(state: typeof chainState.State) {
  const process = state.process_assessment;
  const performance = state.performance_assessment;

  if (!process || !performance) {
    return {};
  }

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const structuredOverview = model.withStructuredOutput(portraitOverviewSchema);
  const processJson = JSON.stringify(process, null, 2);
  const performanceJson = JSON.stringify(performance, null, 2);

  const overviewResult = await structuredOverview.invoke([
    {
      role: 'system',
      content: `你是一位教学诊断专家。请基于提供的过程考查与最终表现结论，提炼一句概括学生当前水平、一句指出关键优势、一句强调优先改进方向，共 2-3 句话，严格返回 JSON。不得引入原数据之外的假设。`,
    },
    {
      role: 'human',
      content: `学习主题或课程: ${state.theme || '未提供'}`,
    },
    {
      role: 'human',
      content: `学生基础信息（用于理解背景）:\n${formatStudentInfo(state.student_info)}`,
    },
    {
      role: 'human',
      content: `过程考查结论(JSON，仅结构化明细):\n${processJson}`,
    },
    {
      role: 'human',
      content: `最终表现结论(JSON，仅结构化明细):\n${performanceJson}`,
    },
    // {
    //   role: 'human',
    //   content: `宏观指标参考：\n- 学习计划与作答摘要（全局指标）：\n${state.plan_summary || '无'}\n- 最终作品解析摘要（全局表现）：\n${state.artifact_summary || '无'}`,
    // },
  ]);

  const portrait: StudentPortrait = {
    overview: overviewResult.overview,
    process_assessment: process,
    performance_assessment: performance,
  };

  return { portrait };
}

async function renderMarkdown(state: typeof chainState.State) {
  const portrait = state.portrait;

  if (!portrait) {
    return { portrait_markdown: '# 学生学习画像\n\n暂无画像结果。' };
  }

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(portraitMarkdownSchema);
  // const portraitJson = JSON.stringify(portrait, null, 2);

  const result = await structuredLlm.invoke([
    {
      role: 'system',
      content: `你是一位教学诊断专家，需要把既定的学生画像内容排版为 Markdown。请严格基于提供的画像 JSON，保持事实一致，输出对象 { "markdown": string }。Markdown 至少包含：
- “总体概览”段落，用于呈现 portrait.overview；
- “评分概览”板块，同时列出 process_assessment.score.total_score/grade 与 performance_assessment.overall_score.total_score/grade；
- “过程考查”“最终表现”两个一级标题，按节点、指标、建议等信息分条说明，可借助列表使结构清晰；
- “下一步建议”或类似标题，整合 process_assessment.next_steps 与 communication_notes、performance_assessment.monitoring_recommendations。
禁止增删事实或新编内容，可对表述做轻微润色以便阅读。`,
    },
    {
      role: 'human',
      content: `学习主题或课程: ${state.theme || '未提供'}`,
    },
    {
      role: 'human',
      content: `学生基础信息（只做背景提示，不要展开重述）:\n${formatStudentInfo(state.student_info)}`,
    },
    {
      role: 'human',
      content: `学生画像:\n${portrait.overview}`,
    },
  ]);

  await uploadPortraitMarkdownToS3(result.markdown, state.student_info);

  return {
    portrait_markdown: result.markdown,
  };
}

const workflow = new StateGraph(chainState)
  .addNode('summarizePlanDetail', summarizePlanDetail)
  .addNode('analyzeArtifacts', analyzeArtifacts)
  .addNode('assessProcess', assessProcess)
  .addNode('assessPerformance', assessPerformance)
  .addNode('generatePortrait', generatePortrait)
  .addNode('renderMarkdown', renderMarkdown)
  .addEdge('__start__', 'summarizePlanDetail')
  .addEdge('__start__', 'analyzeArtifacts')
  .addEdge('summarizePlanDetail', 'assessProcess')
  .addEdge('analyzeArtifacts', 'assessPerformance')
  .addEdge('assessProcess', 'generatePortrait')
  .addEdge('assessPerformance', 'generatePortrait')
  .addEdge('generatePortrait', 'renderMarkdown')
  .addEdge('renderMarkdown', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
