import * as vscode from 'vscode';
import { isSupportedLanguage, getNodeAtFileOffset, findParentFunction } from './utils';
import { createSourceFile, SyntaxKind, ScriptTarget } from 'typescript';

function getEditor(): vscode.TextEditor | null {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !isSupportedLanguage(editor.document.languageId)) {
    vscode.window.showErrorMessage('AST Queries only supported for TypeScript and JavaScript files');
    return null;
  }
  return editor;
}

function addCommentCommand() {
  const editor = getEditor();
  if (!editor) {
    return;
  }

  const ast = createSourceFile(editor.document.fileName, editor.document.getText(), ScriptTarget.Latest, true);
  const node = getNodeAtFileOffset(ast, editor.document.offsetAt(editor.selection.active));
  const parentFunction = findParentFunction(node);
  if (parentFunction) {
    const startPos = editor.document.positionAt(parentFunction.getStart());
    const functionName = parentFunction.name ? parentFunction.name.getText() : 'anonymous function';
    const indent = ' '.repeat(startPos.character);
    editor.edit((editBuilder) => {
      editBuilder.insert(startPos, `/* Comment for ${functionName} */\n${indent}`);
    });
  } else {
    vscode.window.showInformationMessage('Please place the caret inside a function or a method');
  }
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(...[vscode.commands.registerCommand('extension.articulate', addCommentCommand)]);
}

export function deactivate() {}
