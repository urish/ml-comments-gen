import chalk from 'chalk';
import { appendFileSync, createReadStream, writeFileSync } from 'fs';
import { join } from 'path';
import { createInterface } from 'readline';
import { createGunzip } from 'zlib';
import { cleanJsDoc } from './clean-jsdoc';
import { dumpAst } from './dump-ast';
import { renameArgsInComments } from './rename-args-in-comments';

const input = createReadStream(
  join(__dirname, '../../data/typescript-all-functions-100-stars.json.gz')
).pipe(createGunzip());

const datasetPath = join(__dirname, '../../data/dataset.json');
const metadataPath = join(__dirname, '../../data/metadata.txt');
const NEW_LINE = '\r\n';
const N_OBSERVATIONS = 100;
const MAX_COMMENT_LENGTH = 200;
const MAX_AST_LENGTH = 500;

const inputStream = createInterface({ input });

export interface IInputRecord {
  id: string;
  paths: string[];
  line: string;
  character: string;
  comments: string;
  text: string;
}

export interface IProcessedEntry {
  id: string;
  line: number;
  character: number;
  comments: string;
  ast: string;
}

function prepareEntry(input: IInputRecord) {
  const { id, line, character } = input;
  return {
    id,
    line,
    character,
    code: input.text,
    comments: renameArgsInComments(cleanJsDoc(input.comments), input.text),
    ast: dumpAst(input.text, { functionOrMethod: true })
  };
}

console.log(chalk.yellow('Creating dataset. Hold tight!'));
writeFileSync(datasetPath, '');

let allObservations = 0;
let avgCommentLength = 0;
let avgAstLength = 0;
let skipped = 0;
let nExamples = 0;
let maxComment = 0;
let maxAST = 0;

inputStream
  .on('line', (entry) => {
    allObservations++;

    const parsedRecord = JSON.parse(entry);
    const observation = prepareEntry(parsedRecord);

    const commentLength = observation.comments.split(' ').length;
    const astLength = observation.ast.split(' ').length;

    maxComment = Math.max(maxComment, commentLength);
    maxAST = Math.max(maxAST, astLength);

    if (!observation.comments || commentLength > MAX_COMMENT_LENGTH || astLength > MAX_AST_LENGTH) {
      // skip entries without a comment or comments that exceed the max length
      skipped++;
      return;
    }

    nExamples++;
    const outputLine = JSON.stringify(observation) + NEW_LINE;
    appendFileSync(datasetPath, outputLine, { encoding: 'utf-8' });

    // calculate average length over all processed comments
    avgCommentLength = calcSlidingAverage(avgCommentLength, nExamples, commentLength);
    avgAstLength = calcSlidingAverage(avgAstLength, nExamples, astLength);

    if (nExamples >= N_OBSERVATIONS) {
      inputStream.close();
    }
  })
  .on('close', () => {
    const metadata = stripIndent`
      Skipped: ${skipped}
      Average Comment Length: ${avgCommentLength.toFixed(2)}
      Max Comment Length: ${maxComment}/${MAX_COMMENT_LENGTH}
      Average AST Length: ${avgAstLength.toFixed(2)}
      Max AST Length: ${maxAST}
      Observations: ${allObservations}
      Entries in Dataset: ${nExamples}
    `;

    // save metadata to disk
    writeFileSync(metadataPath, metadata);

    console.log(chalk.yellow(metadata));
    console.log(chalk.green('Dataset successfully created.'));
    process.exit(0);
  });

function stripIndent(strings: TemplateStringsArray, ...values: any[]) {
  const endResult = String.raw(strings, ...values);
  const match = endResult.match(/^[ \t]*(?=\S)/gm);

  if (match === null) {
    return endResult;
  }

  const indent = Math.min(...match.map((el) => el.length));
  const regexp = new RegExp('^[ \\t]{' + indent + '}', 'gm');

  return (indent > 0 ? endResult.replace(regexp, '') : endResult).trim();
}

function calcSlidingAverage(curr: number, numExamples: number, exampleLength: number) {
  return (curr * (numExamples - 1)) / numExamples + exampleLength / numExamples;
}
