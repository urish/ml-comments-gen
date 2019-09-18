import * as path from 'path';
import * as fs from 'fs';
import { CommentPredictor } from './comment-predictor';
import { loadModel } from './load-model';

const demoFns = [
  `ngOnDestroy() {
    this.destroy$.next();
  }`,
  `saveCode() {
    const name = prompt('Enter name?');
    if (name) {
      this.nuggetId = name;
      this.nuggetService.save(this.nuggetId, this.code);
      this.router.navigate(['nuggets', this.nuggetId]);
    }
  }`,
  `private logError(title: string, exception: Error) {
		if (console) {
			console.error(title, exception);
		}
  }`,
  `	public observe<T>(query: firebase.database.Query, eventType: firebase.database.EventType = 'value'): Observable<T> {
		return new Observable<T>((observer) => {
			const listener = query.on(
				eventType,
				(snap) => {
					this.zone.run(() => {
						observer.next(snap.val() as T);
					});
				},
				(err: Error) => observer.error(err),
			);

			return () => query.off(eventType, listener);
		}).pipe(
			publish(),
			refCount(),
		);
  }`,
  `  toggle() {
    if (this.instrument.enabled) {
      this.previousVolume = this.instrument.volume;
      this.instrument.volume = 0;
    } else if (this.previousVolume > 0) {
      this.instrument.volume = this.previousVolume;
    }
    this.instrument.enabled = !this.instrument.enabled;
  }`
];
const modelNames = ['js-500', 'js-600', '860', '2000', '2000-trainpy'];
const modelDir = '../runs';
const outputFileName = 'evaluate.json';

async function main() {
  let rows = [];
  rows.push(['function', ...modelNames]);
  const predictors = await Promise.all(
    modelNames.map(async (run) => {
      const { model, tokenizers } = await loadModel(path.join(modelDir, run, 'tfjsmodel'));
      return new CommentPredictor(model, tokenizers);
    })
  );

  for (const fn of demoFns) {
    const row = [fn.trim()];
    for (const predictor of predictors) {
      row.push(Array.from(predictor.predict(fn)).join(''));
    }
    rows.push(row);
  }
  fs.writeFileSync(outputFileName, JSON.stringify(rows, null, 2));
}

main()
  .then(console.log)
  .catch(console.error);
