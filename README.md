# זיקוק ידע לסיווג תאי דם

פרויקט בקורס Deep Learning הבודק האם Knowledge Distillation משפר מסווג קטן לעומת אימון רגיל של אותה ארכיטקטורה.

- נתונים: BloodMNIST עם שמונה מחלקות ותמונות RGB בגודל 224 על 224.
- מורה: DenseNet-121 מאומן מראש על BloodMNIST.
- תלמיד מתוכנן: CNN קטן; השוואה לאימון רגיל ובדיקת טמפרטורות ומשקלי זיקוק.
- מדד ראשי: Balanced Accuracy; בנוסף Accuracy, ניתוח לפי מחלקה, גודל מודל וזמן חיזוי.
- האימונים יבוצעו במחשב של השותפה בעלת ה־GPU; שני השותפים יוכלו לנתח תוצאות.

## מצב הפרויקט

קיימים מחברת חקירת נתונים ראשונית וקובצי הנתונים המקומיים. בסביבה הקיימת נבדקו Python 3.12.7, טעינת ספריות הנתונים, תקינות תלויות וחישובי GPU בסיסיים. טרם אומתו משקולות המורה, וטרם מומשו אימוני התלמיד או מחברת ניתוח עצמאית.

תיעוד ההתקנה שלהלן אינו מפעיל אימונים. התקנה מלאה חדשה בסביבה נקייה עדיין לא נבדקה.

## קובצי התלויות

| קובץ | שימוש |
|---|---|
| `requirements-analysis.txt` | ספריות למחברות ניתוח, קריאת טבלאות ויצירת גרפים; ללא PyTorch ו־CUDA |
| `requirements.txt` | הסביבה המלאה; כוללת את קובץ הניתוח ומוסיפה ספריות נתונים ומודלים |
| `requirements-lock-gpu.txt` | תמונת גרסאות של הסביבה המותקנת ב־Windows עם GPU נכון ל־16 בנובמבר 2026, כולל תלויות עקיפות |

הגרסאות הישירות הקיימות נשמרו. `timm==1.0.31` ו־`safetensors==0.8.0` הותקנו ונבדקו ב־16 בנובמבר 2026, ללא שינוי גרסאות הספריות הקודמות. קובץ הנעילה עודכן וכולל גם אותן. נבדקו יצירת DenseNet-121 עם שמונה פלטים ומשקולות אקראיות, forward ב־GPU על קלט בגודל 1 על 3 על 224 על 224, ושמירה וטעינה מדויקת של tensor באמצעות safetensors ב־CPU וב־GPU. גם pip check ובדיקה מדומה של requirements.txt עברו. הבדיקות אינן אימון ואינן אימות של משקולות המורה שפורסמו; משקולות אלו טרם הורדו ונבדקו. שחזור התקנה מלאה בסביבה נקייה עדיין לא נבדק.

## יצירת סביבה מקומית

הפקודות מיועדות ל־PowerShell ב־Windows, מתוך תיקיית הפרויקט לאחר clone או pull מ־Git. מומלץ להשתמש ב־Python 3.12; גרסת הייחוס שנבדקה היא 3.12.7. מערכת ההפעלה של השותף טרם נבדקה; במערכת אחרת נדרשת התאמת נתיב Python והתקנת PyTorch.

לסביבה חדשה בלבד:

```powershell
py -3.12 -m venv .venv
```

אם `.venv` כבר קיימת, משתמשים בה ולא יוצרים אותה מחדש. הפקודות הבאות פונות ישירות ל־Python שבתוכה, ולכן אין צורך ב־Activate או בשינוי מדיניות PowerShell. ב־VS Code בוחרים את `.venv\Scripts\python.exe` כמפרש וכקרנל המחברת.

## התקנה לשותף לצורך ניתוח תוצאות

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m jupyter lab
```

המסלול מיועד למחברות שיקראו CSV ותוצאות שמורות, בלי לטעון מודלים. הוא אינו מבטיח שכל מחברת בפרויקט תעבוד: `notebooks/01_explore_bloodmnist.ipynb` הקיימת משתמשת ב־medmnist, שדורשת גם torch ו־torchvision. כדי להריץ אותה או קוד שטוען מודלים, יש להתקין את הסביבה המלאה במסלול CPU שלהלן. מחברת ניתוח התוצאות הנפרדת עדיין בתכנון.

## התקנה מלאה במחשב האימון עם NVIDIA

במחשב שנבדק מותקנים RTX 4050 Laptop עם 6GB VRAM, ‏PyTorch 2.13.0+cu130 ו־torchvision 0.28.0+cu130. אם הסביבה הזאת כבר פועלת, אין צורך להתקין את PyTorch מחדש; משלימים את התלויות דרך `requirements.txt`.

בסביבת Windows חדשה עם דרייבר NVIDIA תואם, מתקינים תחילה את גרסת CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

לאחר מכן מתקינים את הסביבה המלאה ומאמתים אותה:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
.\.venv\Scripts\python.exe -c "import timm, safetensors; print('timm:', timm.__version__); print('safetensors:', safetensors.__version__)"
```

מצופה ש־CUDA יהיה זמין במחשב האימון. אם אינו זמין, בודקים מפרש Python, חבילת PyTorch ודרייבר לפני אימון. אין מעבר שקט לאימון CPU. מחוץ למחשב שנבדק יש להתאים את ההתקנה לפי מערכת ההפעלה והדרייבר, ולא להניח שפקודת CUDA אחת מתאימה לכל מחשב.

הגרסאות ב־`requirements.txt` אינן כוללות סיומת CUDA כדי לתמוך גם במסלול CPU. כשגרסת `+cu130` התואמת כבר מותקנת, היא מקיימת את דרישת הגרסה הציבורית. בחירת הבנייה נעשית במפורש בפקודת ההתקנה הראשונה.

## סביבה מלאה ללא GPU

לשותף שירצה להריץ את מחברת חקירת הנתונים או לבדוק קוד מודלים ב־CPU, בסביבת Windows חדשה:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

אין צורך ב־CUDA במסלול הזה. אם מותקנת כבר גרסת GPU ורוצים לעבור ל־CPU, אין להניח שפקודה ללא reinstall תחליף אותה; עדיף לבדוק בסביבה חדשה ונפרדת. אין כוונה לבצע אימונים מלאים במחשב השותף.

## תוצאות משותפות ו־Git

- משתפים CSV, הגדרות, מדדים וגרפים כדי ששניכם תוכלו לנתח את אותן ריצות.
- לכל קובץ תוצאות מצרפים method, seed, מזהה הרצה והגדרות, כדי לשמור על קשר לניסוי המקורי.
- קובצי requirements, README, קוד ומחברות נשמרים ב־Git. תיקיות `.venv`, ‏`data` ו־`checkpoints` נשארות מחוץ ל־Git בהתאם ל־`.gitignore`.
- סביבת Python מקומית לכל שותף; לא מעתיקים `.venv` בין מחשבים.
- אחרי הוספת ספרייה מעדכנים את קובץ התלויות הישירות המתאים. ספרייה שמשמשת ניתוח בלבד נכנסת ל־`requirements-analysis.txt`; תלויות מודלים נכנסות ל־`requirements.txt`.
- לא מחליפים את רשימות ההתקנה לשותף ב־pip freeze ממחשב ה־GPU: הוא כולל תלויות חומרה ותלויות עקיפות שאינן דרושות לניתוח.

## שחזור תמונת סביבת ה־GPU

`requirements-lock-gpu.txt` הוא snapshot מהסביבה הקיימת, ולא קובץ התקנה לשותף או הוכחה להתקנה נקייה שנבדקה. הוא מכיל חבילות ייעודיות ל־Windows וגרסאות torch עם סיומת `+cu130`.

לשחזור תמונת המצב בסביבת Windows חדשה ונפרדת, מתקינים קודם את PyTorch ו־torchvision ממקור cu130 כמתואר לעיל, ואז:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-gpu.txt
.\.venv\Scripts\python.exe -m pip check
```

המסלול משחזר את רשימת הגרסאות הנוכחית בלבד, כולל timm ו־safetensors שנבדקו מקומית. לאחר שינוי נוסף בספריות יש להפיק snapshot חדש ולתעד מה אומת. הורדת משקולות, בדיקת המורה ואימון הם שלבים נפרדים מהתקנת ספריות.

## מקורות התקנה

- [פורמט קובצי requirements והכללת קובץ נוסף](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
- [התקנת PyTorch לפי מערכת הפעלה ופלטפורמת חישוב](https://pytorch.org/get-started/locally/)
- [גרסאות timm ב־PyPI](https://pypi.org/project/timm/)
- [גרסאות safetensors ב־PyPI](https://pypi.org/project/safetensors/)
