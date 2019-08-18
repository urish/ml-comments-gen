import { renameArgsInComments } from './rename-args-in-comments';

describe('renameArgsInComments', () => {
  it('should rename args found in the given comments', () => {
    const src = `function(a, b) => a + b`;
    expect(renameArgsInComments('/* sum(a,b): should return the sum of a and b */', src)).toBe(
      '/* sum($__arg0$,$__arg1$): should return the sum of $__arg0$ and $__arg1$ */',
    );
  });

  it('should also rename args for methods', () => {
    const src = `methodName(foo, bar) => foo * bar`;
    expect(renameArgsInComments('/* multiplies foo by bar */', src)).toBe('/* multiplies $__arg0$ by $__arg1$ */');
  });

  it('should not fail for methods with no arguments', () => {
    const src = `methodName() => foo * bar`;
    expect(renameArgsInComments('/* i have no arguments */', src)).toBe('/* i have no arguments */');
  });
});
