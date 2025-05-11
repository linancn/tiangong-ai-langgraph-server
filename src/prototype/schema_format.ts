import $RefParser from '@apidevtools/json-schema-ref-parser';
import { writeFileSync } from 'fs';

const input = 'test_data/process_schema1.json'
const output = 'test_data/process_schema1_full';

async function dereferenceSchema(schemaPath: string): Promise<void> {
  try {
    // 解析JSON Schema并解决所有引用
    const schema = await $RefParser.dereference(schemaPath);
    
    // 保存未压缩版本
    const jsonString = JSON.stringify(schema, null, 2);
    writeFileSync(output+'.json', jsonString);
    
    // 保存没有空格和换行的压缩版本
    const minifiedJson = JSON.stringify(schema);
    writeFileSync(output+'min.json', minifiedJson);
    
    console.log('Schema处理完成，已生成完整版和压缩版');
  } catch (err) {
    console.error('处理Schema时出错:', err);
  }
}

// 调用函数
dereferenceSchema(input);

