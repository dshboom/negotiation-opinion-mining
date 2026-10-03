import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'src/build_structured_tasks.py'
class StructuredTasksTest(unittest.TestCase):
    def test_document_groups_and_task_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[dict(sample_id=str(i),docs=[dict(full_text='same' if i<2 else str(i))],issue_list=[dict(argument_chain=['evidence'],issue_name='topic',stance='support')],future_argument=['future']) for i in range(8)]
            source=root/'train.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            def build(out):
                subprocess.run([sys.executable,str(SCRIPT),'--train',str(source),'--out',str(out),'--holdout','3'],check=True,capture_output=True)
            a=root/'a';b=root/'b';build(a);build(b)
            ma=json.loads((a/'split_manifest.json').read_text());mb=json.loads((b/'split_manifest.json').read_text())
            self.assertEqual(ma,mb)
            fit=set(ma['fit_ids']);held=set(ma['holdout_ids'])
            self.assertFalse(fit&held)
            self.assertEqual('0' in held,'1' in held)
            self.assertEqual(len(fit|held),8)
            ex=[json.loads(x) for x in (a/'fit.extraction.jsonl').read_text(encoding='utf-8').splitlines()]
            answer=json.loads(ex[0]['messages'][-1]['content'])
            self.assertEqual(list(answer['issue_list'][0]),['argument_chain','issue_name','stance'])
            self.assertNotIn('future_argument',answer)
            futures=[json.loads(x) for x in (a/'fit.future.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(r['source_sample_id'] in fit for r in futures))
if __name__=='__main__':unittest.main()
