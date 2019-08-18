import { cleanJsDoc } from './clean-jsdoc';

describe('cleanJsdoc', () => {
  it('should remove jsdoc lines, but keep @param and @returns', () => {
    const comment = `/**
      * Adds a new account
      * @class Account
      * @method create
      * @param name The name of the account
      * @param balance The account balance
      * @returns nothing!
      */`;
    expect(cleanJsDoc(comment)).toBe(`/**
      * Adds a new account
      * @param name The name of the account
      * @param balance The account balance
      * @returns nothing!
      */`);
  });

  it('should remove html tags', () => {
    const comment = `/**
       * Combines multiple Observables to create an Observable whose values are
       * calculated from the latest values of each of its input Observables.
       *
       * <span class="informal">Whenever any input Observable emits a value, it
       * computes a formula using the latest values from all the inputs, then emits
       * the output of that formula.</span>
       */`;
    expect(cleanJsDoc(comment)).toBe(`/**
       * Combines multiple Observables to create an Observable whose values are
       * calculated from the latest values of each of its input Observables.
       *
       * Whenever any input Observable emits a value, it
       * computes a formula using the latest values from all the inputs, then emits
       * the output of that formula.
       */`);
  });


  it('should remove code examples from comments', () => {
    const comment = `/**
      * ### Example
      *
      * \`\`\`
      * @Component({
      *   viewProviders: [
      *     IterableDiffers.extend([new ImmutableListDiffer()])
      *   ]
      * })
      * \`\`\`
      *
      * Easy peasy!
      */`;
    expect(cleanJsDoc(comment)).toBe(`/**
      * ### Example
      *
      *
      * Easy peasy!
      */`);
  });
});
