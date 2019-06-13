import chalk from 'chalk';
import { appendFileSync, writeFileSync, createReadStream } from 'fs';
import { join } from 'path';
import { createInterface } from 'readline';
import { createGunzip } from 'zlib';
import { dumpAst } from './dump-ast';

const input = createReadStream(join(__dirname, '../../data/typescript-all-functions.json.gz')).pipe(createGunzip());
const datasetPath = join(__dirname, '../../data/dataset.json');
const NEW_LINE = '\r\n';
const N_OBSERVATIONS = 1000;

const inputStream = createInterface({ input });

export interface IInputRecord {
  id: string;
  paths: string[];
  line: string;
  character: string;
  text: string;
}

export interface IProcessedEntry {
  id: string;
  line: number;
  character: number;
  ast: string;
}

function prepareEntry(input: IInputRecord) {
  const { id, line, character } = input;
  return {
    id,
    line,
    character,
    ast: dumpAst(input.text, true).map((e) => e.replace(/~/g, '')),
  };
}

console.log(chalk.yellow('Creating dataset. Hold tight!'));
writeFileSync(datasetPath, '');

let processedEntries = 0;

inputStream
  .on('line', (entry) => {
    const parsedRecord = JSON.parse(entry);
    const observation = prepareEntry(parsedRecord);

    const outputLine = JSON.stringify(observation) + NEW_LINE;
    appendFileSync(datasetPath, outputLine, { encoding: 'utf-8' });

    if (++processedEntries >= N_OBSERVATIONS) {
      inputStream.close();
    }
  })
  .on('close', () => {
    console.log(chalk.green('Dataset successfully created.'));
    process.exit(0);
  });
