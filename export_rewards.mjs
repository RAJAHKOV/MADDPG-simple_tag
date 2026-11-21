import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const data = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const { agents, rewards, interval, output } = data;
const count = rewards[agents[0]].length;
const col = number => {
  let name = '';
  for (let n = number; n > 0; n = Math.floor((n - 1) / 26)) {
    name = String.fromCharCode(65 + (n - 1) % 26) + name;
  }
  return name;
};
const endAgent = col(agents.length + 1);
const end = col(agents.length + 2);
const headers = ['episode', ...agents, 'sum reward'];
const workbook = Workbook.create();
const consoleSheet = workbook.worksheets.add('终端记录');
const detail = workbook.worksheets.add('每回合奖励');
const selected = [];
for (let episode = interval; episode <= count; episode += interval) selected.push(episode);

detail.getRange(`A1:${end}1`).values = [headers];
const raw = Array.from({ length: count }, (_, i) => [i + 1, ...agents.map(a => rewards[a][i])]);
detail.getRange(`A2:${endAgent}${count + 1}`).values = raw;
detail.getRange(`${end}2:${end}${count + 1}`).formulas = raw.map((_, i) => [`=SUM(B${i + 2}:${endAgent}${i + 2})`]);
const sourceCol = col(agents.length + 4);
detail.getRange(`${sourceCol}1`).values = [[`来源：${data.source}`]];
detail.getRange(`${sourceCol}2`).values = [['奖励为每回合累计值；未四舍五入写入。']];
detail.getRange(`${sourceCol}1:${sourceCol}2`).format.columnWidth = 72;

consoleSheet.getRange('A2').values = [['训练奖励记录']];
consoleSheet.getRange('A2').format.font = { name: 'Arial', size: 16, bold: true, color: '#183153' };
consoleSheet.getRange('A3').values = [[`每 ${interval} 回合取该回合累计奖励，与终端打印方式一致，不是区间平均。`]];
consoleSheet.getRange(`A5:${end}5`).values = [headers];
if (selected.length) {
  consoleSheet.getRange(`A6:${end}${selected.length + 5}`).formulas = selected.map(episode =>
    headers.map((_, i) => `='每回合奖励'!${col(i + 1)}${episode + 1}`));
}

for (const [sheet, head, rows] of [[consoleSheet, 5, selected.length], [detail, 1, count]]) {
  sheet.showGridLines = false;
  sheet.getRange(`A${head}:${end}${head + rows}`).format.font = { name: 'Arial', size: 11, color: '#18283D' };
  sheet.getRange(`A${head}:${end}${head + rows}`).format.rowHeight = 21;
  sheet.getRange(`A${head}:${end}${head}`).format = {
    fill: '#233C60', font: { name: 'Arial', size: 11, bold: true, color: '#FFFFFF' },
    horizontalAlignment: 'center', verticalAlignment: 'center', rowHeight: 27,
  };
  sheet.getRange(`A${head}:A${head + rows}`).format.columnWidth = 13;
  sheet.getRange(`B${head}:${end}${head + rows}`).format.columnWidth = 21;
  if (rows) {
    sheet.getRange(`A${head + 1}:A${head + rows}`).setNumberFormat('0');
    sheet.getRange(`B${head + 1}:${end}${head + rows}`).setNumberFormat('0.000000');
    sheet.getRange(`A${head + 1}:${end}${head + rows}`).format.horizontalAlignment = 'right';
    sheet.tables.add(`A${head}:${end}${head + rows}`, true,
      sheet === detail ? 'AllEpisodeRewards' : 'ConsoleEpisodeRewards');
  }
  sheet.freezePanes.freezeRows(head);
}

workbook.recalculate();
// Check source values, linked rows and formula results at first/middle/last positions.
for (const i of new Set([0, Math.floor(count / 2), count - 1])) {
  const row = detail.getRange(`A${i + 2}:${end}${i + 2}`).values[0];
  assert.deepEqual(row.slice(0, -1), raw[i]);
  const expected = agents.reduce((sum, a) => sum + rewards[a][i], 0);
  assert.ok(Math.abs(row.at(-1) - expected) < 1e-9);
}
for (const i of new Set([0, Math.floor(selected.length / 2), selected.length - 1])) {
  if (i < 0 || !selected.length) continue;
  const row = consoleSheet.getRange(`A${i + 6}:${end}${i + 6}`).values[0];
  assert.equal(row[0], selected[i]);
  assert.deepEqual(row.slice(1, -1), agents.map(a => rewards[a][selected[i] - 1]));
}
console.log((await workbook.inspect({ kind: 'table', range: `终端记录!A5:${end}9`,
  include: 'values,formulas', tableMaxRows: 5, tableMaxCols: headers.length, maxChars: 2400 })).ndjson);
const errors = await workbook.inspect({ kind: 'match',
  searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 10 }, maxChars: 1200,
  summary: 'Formula error scan' });
console.log(errors.ndjson);
if (data.previewDir) {
  await fs.mkdir(data.previewDir, { recursive: true });
  for (const [sheetName, range, name] of [
    ['终端记录', `A1:${end}16`, 'console'], ['每回合奖励', `A1:${end}12`, 'detail']]) {
    const preview = await workbook.render({ sheetName, range, scale: 1.5, format: 'png' });
    await fs.writeFile(path.join(data.previewDir, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}
await fs.mkdir(path.dirname(output), { recursive: true });
await (await SpreadsheetFile.exportXlsx(workbook)).save(output);
console.log(`PASS: ${count} full episodes; ${selected.length} console rows`);
