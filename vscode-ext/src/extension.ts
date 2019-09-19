import * as vscode from 'vscode';
import { isSupportedLanguage, getNodeAtFileOffset, findParentFunction, getNodeAtCursor } from './utils';
import { createSourceFile, ScriptTarget } from 'typescript';
import { loadModel, CommentPredictor } from 'ts-comment-predictor';
import { modelPath } from 'ts-comment-predictor-model';

let predictorPromise: Promise<CommentPredictor>;

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

  const node = getNodeAtCursor(editor);
  const parentFunction = findParentFunction(node);
  if (parentFunction) {
    const startPos = editor.document.positionAt(parentFunction.getStart());
    const indent = ' '.repeat(startPos.character);
    const predictor = await predictorPromise;
    await editor.edit((editBuilder) => {
      editBuilder.insert(startPos, '/* */\n' + indent);
    });
    let currentPos = new vscode.Position(startPos.line, startPos.character + 1);
    for (let token of predictor.predict(parentFunction.getText())) {
      if (token.startsWith('*/')) {
        continue;
      }
      if (token.startsWith('/*')) {
        token = token.substr(2);
        if (!token.trim()) {
          continue;
        }
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
  predictorPromise = loadModel(modelPath).then(
    ({ model, tokenizers }) => new CommentPredictor(model, tokenizers),
  );
  context.subscriptions.push(...[vscode.commands.registerCommand('extension.addComment', addCommentCommand)]);
}

export function deactivate() {}
