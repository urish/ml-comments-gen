import { renameArgsInComments } from './rename-args-in-comments';

describe('renameArgsInComments', () => {
  it('should rename args found in the given comments', () => {
    const src = `function(a, b) { return a + b; }`;
    expect(renameArgsInComments('/* sum(a,b): should return the sum of a and b */', src)).toBe(
      '/* sum(ArgumentNumber0,ArgumentNumber1): should return the sum of ArgumentNumber0 and ArgumentNumber1 */',
    );
  });

  it('should also rename args for methods', () => {
    const src = `methodName(foo, bar) { return foo * bar }`;
    expect(renameArgsInComments('/* multiplies foo by bar */', src)).toBe(
      '/* multiplies ArgumentNumber0 by ArgumentNumber1 */',
    );
  });

  it('replace the function/method name with a placeholder', () => {
    const src = `takeFive() { return 5 }`;
    expect(renameArgsInComments('/* the best takeFive implementation */', src)).toBe(
      '/* the best FunctionNamePlaceholder implementation */',
    );
  });


  it('should not fail for methods with no arguments', () => {
    const src = `methodName() => foo * bar`;
    expect(renameArgsInComments('/* i have no arguments */', src)).toBe('/* i have no arguments */');
  });
});
