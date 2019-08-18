import { tsquery } from '@phenomnomnominal/tsquery';
import { FunctionDeclaration, MethodDeclaration } from 'typescript';

export function functionOrMethodAST(source: string) {
  const functionNode = tsquery.query<FunctionDeclaration>(source, 'SourceFile>FunctionDeclaration')[0];
  if (functionNode) {
    return functionNode;
  }
  return tsquery.query<MethodDeclaration>(
    `class __DUMMY { ${source} }`,
    'SourceFile>ClassDeclaration>MethodDeclaration',
  )[0];
}
