import * as vscode from 'vscode';
import { isSupportedLanguage, findParentFunction, getNodeAtCursor } from './utils';
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

    vscode.window.withProgress({
      title: 'Predicting comment...',
      location: vscode.ProgressLocation.Notification,
    }, async () => {
      const predictor = await predictorPromise;
      const comment = predictor.predict(parentFunction.getText());
      editor.edit((editBuilder) => {
        editBuilder.insert(startPos, comment + '\n' + indent);
      });
    });
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
