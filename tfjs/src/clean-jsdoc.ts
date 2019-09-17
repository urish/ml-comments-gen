import * as striptags from 'striptags';

export function cleanJsDoc(comments: string) {
  return striptags(comments)
    .replace(/(^|\r?\n)[ \t]*\*[ \t]*```.+\r?\n[ \t]*\*[ \t]*```/gs, '')
    .replace(/(^|\r?\n)[ \t]*\*[ \t]*@(\S+).+/g, (match, _, name) =>
      ['param', 'return', 'returns'].includes(name) ? match : '',
    );
}
