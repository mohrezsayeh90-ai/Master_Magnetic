"""Build a self-contained Kaggle notebook; never package credentials or local config."""
from pathlib import Path
import base64,io,zipfile,json
ROOT=Path(__file__).resolve().parents[1]

def build():
    buffer=io.BytesIO()
    included=[]
    for pattern in ('magnetic_control/*.py','checkpoints/*.pt','scripts/train_policy.py','scripts/validate.py','examples/*.csv'):
        included.extend(ROOT.glob(pattern))
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(included):z.write(path,path.relative_to(ROOT))
    payload=base64.b64encode(buffer.getvalue()).decode()
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))
    def code(text):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True)))
    md('# کنترل مسیر ذره با سیاست معکوس آموزش‌دیده\n\nبرگرفته از Master8. مسیر فضایی با نقاط `z_m,x_m` داده می‌شود؛ کنترل‌کننده از میدان‌های آموزش‌دیده و بازخورد موقعیت استفاده می‌کند. زمان‌بندی طولی خروجی مدل است. خروجی `u` تحریک مؤثر دو منبع آینه‌ای است، نه جریان کالیبره‌شدهٔ سیم‌پیچ.\n\nکد و گزارش‌ها: https://github.com/mohrezsayeh90-ai/Master_Magnetic\n\nاین نوت‌بوک بدون توکن و اینترنت اجرا می‌شود. وزن‌ها داخل فایل هستند؛ اجرای مجدد، پیش‌فرض آموزش مجدد نمی‌دهد.')
    code("import base64,io,zipfile,sys\nfrom pathlib import Path\nROOT=Path.cwd()/'magnetic_controller'\nROOT.mkdir(exist_ok=True)\nPAYLOAD='"+payload+"'\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:\n    archive.extractall(ROOT)\n(ROOT/'results').mkdir(exist_ok=True)\nsys.path.insert(0,str(ROOT))\nprint('Controller extracted:',ROOT)\n")
    md('## آموزش اختیاری\nوزن سیاست جدید از ۶۰هزار حالت آموزش دیده و روی ۱۲هزار حالت مستقل آزموده شده است. برای بازآموزی `TRAIN_AGAIN=True` کنید. ثابت‌های فیزیکی از وزن‌های Master8 خوانده می‌شوند؛ تغییرشان نیازمند آموزش میدان‌ها است.')
    code("TRAIN_AGAIN=False\nif TRAIN_AGAIN:\n    import runpy\n    sys.argv=['train_policy.py','--steps','2500']\n    runpy.run_path(str(ROOT/'scripts/train_policy.py'),run_name='__main__')\n")
    md('## مسیر ورودی\nمسیر پیش‌فرض نمونه است. برای مسیر خود `PATH_CSV` را به فایل دارای عنوان `z_m,x_m` تغییر دهید. حداقل چهار نقطه با z صعودی، شروع `(z,x)=(-0.08,0)` متر، پایان z حداکثر 0.072 متر و |x| کمتر از 52.5 میکرومتر لازم است. برای مسیر زمان‌دار، `MODE="time"` و CSV با عنوان `time_s,x_m` بدهید؛ افق حداکثر 5.0666667 ثانیه است. شدنی‌بودن مسیر بعد از شبیه‌سازی بررسی می‌شود.')
    code("import numpy as np,json\nfrom magnetic_control.runtime import load_fields\nfrom magnetic_control.tracking import Target,track\nfrom magnetic_control.export import save_result\nPATH_CSV=ROOT/'examples/bump.csv'\nMODE='spatial'\ncfg=load_fields()\ntarget=Target(np.loadtxt(PATH_CSV,delimiter=',',skiprows=1),cfg,MODE)\nresult=track(target,cfg)\nsave_result(result,cfg,ROOT/'results/custom')\nprint(json.dumps(result['report'],indent=2))\nif not result['report']['accepted']:\n    raise RuntimeError('Target rejected: tracking error, actuation bound or model domain exceeded.')\n")
    code("import matplotlib.pyplot as plt\nfig,axs=plt.subplots(1,2,figsize=(11,4))\naxs[0].plot(result['q'][:,1]*cfg.d*1000,result['target']*cfg.Rv*1e6,'--',label='Target')\naxs[0].plot(result['q'][:,1]*cfg.d*1000,result['q'][:,0]*cfg.Rv*1e6,label='Controlled')\naxs[0].set(xlabel='z (mm)',ylabel='x (micrometres)');axs[0].legend()\naxs[1].plot(result['tau']*cfg.time_scale,result['u']);axs[1].set(xlabel='time (s)',ylabel='effective magnetic command u')\nplt.show()\n")
    md('## آزمون مستقل\nمرجع تحلیلی در آموزش سیاست استفاده نمی‌شود. آزمون‌ها شامل مسیرهای جدید، بازپخش فرمان، افزایش ۱۰٪ جریان و رد مسیر غیرقابل‌اجرا هستند. آستانه ۰٫۷۵ میکرومتر است. این نتیجه فقط مربوط به مدل دوبعدی نظری با حسگر موقعیت ایده‌آل است؛ سخت‌افزار، نویز، تأخیر و نرخ تغییر منبع آزمایش نشده‌اند.')
    code("import runpy\nrunpy.run_path(str(ROOT/'scripts/validate.py'),run_name='__main__')\nprint('Validation:',json.loads((ROOT/'results/validation.json').read_text())['passed'])\nfrom IPython.display import Image,display\ndisplay(Image(filename=str(ROOT/'results/tracking.png')))\n")
    n=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},nbformat=4,nbformat_minor=4)
    path=ROOT/'notebooks/magnetic_path_controller.ipynb';path.write_text(json.dumps(n,ensure_ascii=False,indent=1))
    print(path)
if __name__=='__main__':build()
