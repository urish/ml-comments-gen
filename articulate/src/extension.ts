import * as vscode from 'vscode';
import { isSupportedLanguage, getNodeAtFileOffset, findParentFunction } from './utils';
import { createSourceFile, ScriptTarget } from 'typescript';
import { CommentPredictorStub } from './comment-predictor-stub';

function getEditor(): vscode.TextEditor | null {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !isSupportedLanguage(editor.document.languageId)) {
    vscode.window.showErrorMessage('AST Queries only supported for TypeScript and JavaScript files');
    return null;
  }
  return editor;
}

async function addCommentCommand() {
  const editor = getEditor();
  if (!editor) {
    return;
  }

  const ast = createSourceFile(editor.document.fileName, editor.document.getText(), ScriptTarget.Latest, true);
  const node = getNodeAtFileOffset(ast, editor.document.offsetAt(editor.selection.active));
  const parentFunction = findParentFunction(node);
  if (parentFunction) {
    const startPos = editor.document.positionAt(parentFunction.getStart());
    const indent = ' '.repeat(startPos.character);
    const cp = new CommentPredictorStub();
    await editor.edit((editBuilder) => {
      editBuilder.insert(startPos, '/* */\n' + indent);
    });
    let currentPos = new vscode.Position(startPos.line, startPos.character + 3);
    for (let token of cp.predict(parentFunction.getText())) {
      if (token === '/* ' || token === '*/ ') {
        continue;
      }
      if (token[0] === '\n') {
        token += indent;
      }
      await editor.edit((editBuilder) => {
        editBuilder.insert(currentPos, token);
      });
      if (token[0] === '\n') {
        currentPos = new vscode.Position(currentPos.line + 1, token.length - 1);
      } else {
        currentPos = new vscode.Position(currentPos.line, currentPos.character + token.length);
      }
    }
  } else {
    vscode.window.showInformationMessage('Please place the caret inside a function or a method');
  }
}

export async function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(...[vscode.commands.registerCommand('extension.articulate', addCommentCommand)]);
}

export function deactivate() {}
