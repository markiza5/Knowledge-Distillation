# דוח אימות מורה BloodMNIST

תאריך האימות: 16 בנובמבר 2026.

המורה לזיקוק הוא checkpoint יחיד שנבחר מראש: seed 71397589. שתי גרסאות המשקולות האחרות נבדקו לצורך שחזור דיווח השונות של המחברים בלבד. לא בוצע ensemble, לא נבחר מורה לפי test ולא בוצע אימון מורה או תלמיד.

## תוצאות

הערכים בטבלה הם אחוזים. כל גרסה הוערכה על 1,712 תמונות validation ועל 3,421 תמונות test.

| seed | תפקיד | Validation Balanced Accuracy | Test Accuracy | Test Balanced Accuracy | Test Balanced Accuracy במקור |
|---|---|---|---|---|---|
| 71397589 | מורה הזיקוק הקבוע | 98.6796% | 98.4215% | 98.2223% | 98.2223% |
| 133560673 | בדיקת ייחוס בלבד | 99.0994% | 98.3631% | 98.5521% | 98.5521% |
| 265017005 | בדיקת ייחוס בלבד | 99.4718% | 98.6261% | 98.6525% | 98.6525% |

ממוצע Test Balanced Accuracy של שלוש ריצות המקור: 98.4756%; סטיית תקן מדגמית: 0.2251 נקודות אחוז. התוצאה משחזרת את 98.48% ± 0.23 שפורסמו לאחר עיגול. היא אינה תוצאה של אימון שלוש פעמים אצלנו.

## מה אומת

- קובץ הנתונים המקומי תואם ל־MD5 הרשמי של MedMNIST בגודל 224; SHA256 נשמר בדוח JSON.
- RGB בגודל 224, המרה מ־uint8 לטווח 0 עד 1, ואז נרמול לפי config המורה. אין augmentation בהערכה.
- טעינה מחמירה של כל משקולות DenseNet-121 בן שמונה פלטים מתוך safetensors. אין טעינה חלקית או השלמת שכבות אקראיות.
- eval, פרמטרים קפואים ו־inference_mode. כל state_dict נבדק ללא שינוי לפני ואחרי ההערכה, כולל buffers של BatchNorm, ולא נוצרו gradients.
- inference ב־float32 על RTX 4050 Laptop, עם batch size של 32. אין שימוש ב־AMP או באופטימייזר.
- ששת מבחני היחידה עברו; לכל קובץ תחזיות נבדקו סדר האינדקסים, סכום ההסתברויות והתאמת המדדים בחישוב חוזר.
- תוצאות שלוש הגרסאות תואמות לתוצאות ה־Balanced Accuracy שבקובצי המקור; גרסאות מקור ו־SHA256 נשמרו.

## שחזור ובדיקת תוצאות

מתוך תיקיית הפרויקט, עם הסביבה המלאה ועם הנתונים והמשקולות המקומיים:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B src/verify_teacher.py --output results/teacher_recheck
```

הפקודה האחרונה מבצעת הערכה בלבד. בוחרים תיקיית פלט חדשה לכל בדיקה; אין דריסה אוטומטית של דוח סופי קיים. אין צורך להריץ אותה שוב כדי לנתח את קובצי ה־CSV שכבר נשמרו.

המשקולות נשמרות בתיקיות checkpoints/teachers/seed_<seed>. בכל תיקייה נמצאים config.json, model.safetensors, כרטיס המודל ו־manifest.json. התיקיות מוחרגות מ־Git. הקובץ configs/teachers.json מתעד בנפרד את המקורות, ה־revisions וה־hashes, וצריך להיכנס ל־Git לצד הקוד, הבדיקות ודוחות התוצאות.

לשחזור הורדת המשקולות אפשר להריץ את קטע Python הבא מהשורש באמצעות Python של הסביבה. הוא מוריד קבצים ציבוריים מ־revision קבוע, מאמת SHA256 ואינו מפעיל קוד מהמאגר:

```python
from pathlib import Path
import hashlib
import json
from huggingface_hub import hf_hub_download

manifest = json.loads(Path('configs/teachers.json').read_text(encoding='utf-8'))
for teacher in manifest['teachers']:
    destination = Path('checkpoints/teachers') / f"seed_{teacher['seed']}"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, metadata in teacher['files'].items():
        path = Path(hf_hub_download(teacher['repo_id'], filename, revision=teacher['revision'], local_dir=str(destination), token=False))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256']
    (destination / 'manifest.json').write_text(json.dumps(teacher, indent=2), encoding='utf-8')
```

## מגבלות והבחנות

- אומת המורה המאומן שפורסם; לא שוחזר תהליך אימונו.
- המורה הקבוע לזיקוק הוא seed 71397589, אף שלגרסה אחרת תוצאת test גבוהה יותר. הבחירה לא משתנה בעקבות הבדיקה.
- בכרטיסי המודל מופיעה גם סדרת מספרים אחרת בתוך אזהרת מיצוע משקולות, בלי זיהוי מדד מספיק ברור. לא השתמשנו בה כיעד שחזור; הטבלה כאן משווה לערכי test_balanced_accuracy_per_seed המפורשים ב־config.json.
- מיפוי המחלקות הוא המיפוי הרשמי של BloodMNIST; ההתאמה המספרית למקור מחזקת את אימות מסלול הקלט והפלט.
- שמות splits ובדיקת checksum אינם הוכחה להפרדת מטופלים או להיעדר כפילויות בנתוני המקור.
- זמני הערכה בדוח JSON כוללים הכנת batches ואינם benchmark של latency. גודל מודל וזמני חיזוי יימדדו בהמשך בפרוטוקול נפרד.
- היחס לבדיקה זו הוא אימות של מורה קבוע; בשלב הבא אין להתחיל אימון תלמיד ללא מעבר מפורש לשלב הבא בשיחה.

## מקורות

- [כרטיס המורה הקבוע](https://huggingface.co/sdoerrich97/colorist_densenet121_bloodmnist_s71397589)
- [קוד המקור שנבדק בגרסה קבועה](https://github.com/sdoerrich97/colorist/tree/9c4633d5d530c17ebf46bd0843227491e64f67a0)
- [מקור נתוני BloodMNIST-224](https://zenodo.org/records/10519652)
