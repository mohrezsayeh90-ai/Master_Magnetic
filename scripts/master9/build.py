from pathlib import Path
import zipfile,io,base64,json
ROOT=Path(__file__).resolve().parents[2]
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((ROOT/'master9').glob('*.py')):z.write(p,p.relative_to(ROOT))
 for name in ('checkpoint.pt','training.json','training_progress.json'):
  p=ROOT/'results/master9'/name
  if p.exists():z.write(p,'pretrained/'+name)
 p=ROOT/'examples/master9_target.csv'
 if p.exists():z.write(p,'examples/master9_target.csv')
blob=base64.b64encode(buf.getvalue()).decode();cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)})
def code(s):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]})
md('''# master9 — Parametric inertial particle PINN (GPU)

این نسخه با نسخهٔ قبلیِ کنترل نظارت‌شده فرق دارد: **هیچ برچسب فرمان و هیچ مسیر حل‌شده‌ای وارد آموزش نمی‌شود.** آموزش از باقیماندهٔ معادلات سیال و ذره است؛ حل مستقل Radau فقط بعد از آموزش اجرا می‌شود.

ورودی‌ها: رینولدز، سرعت متوسط سیال، چگالی سیال، محل رهایش، دو مؤلفهٔ سرعت اولیهٔ ذره و تحریک مغناطیسی. ویسکوزیته از `eta=rho*U*(2R)/Re` به دست می‌آید و ترکیب نامعتبر رد می‌شود.

محدوده‌ها: eta=0.001..0.006 Pa.s، U=0.005..0.03 m/s، rho=950..1100 kg/m³، X0=-0.7..0.7، Z0=-2..1.7، vx0=±0.001 m/s، vz0=0..0.06 m/s، u=±0.5. شعاع ذره 500nm و کانال 75µm ثابت‌اند.

معادلات: جریان کاملاً توسعه‌یافتهٔ نیوتنی در کانال استوانه‌ای، میدان تحلیلی دو منبع آینه‌ای و حرکت ذره با درگ استوکس و جرم افزودهٔ نصف جرم سیال جابه‌جاشده. گرانش، نیروی تاریخچهٔ Basset، برهم‌کنش دوطرفه و نویز حسگر لحاظ نشده‌اند. گذرای زیرمیکروثانیه‌ای فقط در محدودهٔ این مدل ساده اعتبارسنجی می‌شود.

ساختار شبکه شرط مکان و سرعت اولیه را دقیقاً اعمال می‌کند و لایهٔ گذرای نمایی را از بخش کند حرکت جدا می‌کند. PINN نگاشت گام زمانی کوتاه را می‌آموزد و برای مسیر بلند تکرار می‌شود. کنترل فرمان، حل معکوس مقید با همین PINN است، نه شبکهٔ دارای برچسب جواب.
''')
code("import os,sys,io,base64,zipfile,json,time\nfrom pathlib import Path\nimport torch\nassert torch.cuda.is_available(), 'GPU was requested but is unavailable; do not silently train on CPU.'\nROOT=Path.cwd()/'master9_run'\nROOT.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode('"+blob+"'))) as z:z.extractall(ROOT)\nsys.path.insert(0,str(ROOT))\nOUTPUT=ROOT/'outputs'\nprint('GPU:',torch.cuda.get_device_name(0))\n")
md('## وزن‌های آموزش‌دیده روی GPU؛ فقط باقیماندهٔ معادلات\nنسخهٔ ۱ روی Tesla T4 با ۶هزار گام سیال و ۱۵هزار گام ذره آموزش دید. وزن‌ها داخل فایل‌اند و پیش‌فرض بارگذاری می‌شوند. برای تکرار آموزش TRAIN_FROM_SCRATCH=True کنید. زمان و GPU ثبت‌شده مربوط به آموزش اصلی است؛ آزمون‌ها با seed جدا و حل مستقل تکرار می‌شوند.')
code("from master9.train import train,load\nTRAIN_FROM_SCRATCH=False  # GPU training already completed in version 1; set True to repeat.\nif TRAIN_FROM_SCRATCH:\n    model,runtime=train(OUTPUT,flow_steps=6000,particle_steps=15000,batch=2048,require_gpu=True)\nelse:\n    import shutil\n    OUTPUT.mkdir(exist_ok=True)\n    for p in (ROOT/'pretrained').iterdir():shutil.copyfile(p,OUTPUT/p.name)\n    model,runtime=load(OUTPUT)\nprint(json.dumps(runtime,indent=2))\n")
md('## آزمون فیزیکی برای هر ورودی\nشامل رینولدز و ویسکوزیتهٔ سازگار، پروفایل و دبی سیال، محل رهایش، سرعت اولیه و میرایی آن، تغییر چگالی، اعتبار درگ استوکس، باقیماندهٔ مستقل و مقایسه با حل stiff Radau است. شکست هر معیار اجرای موفق را متوقف می‌کند؛ آستانه‌ها از قبل در کد تعیین شده‌اند.')
code("from master9.validate import run\nphysics_report=run(OUTPUT,require_gpu=True)\nprint(json.dumps(physics_report,indent=2))\n")
md('## دنبال‌کردن مسیر با پارامترهای جدید\nکنترل به u=±0.5 محدود است. فرمان‌های حاصل در حل مستقل اینرسی‌دار بازپخش می‌شوند. چهار آزمایش شامل سرعت سیال و ویسکوزیتهٔ مرزی و رهایش متغیر است.')
code("from master9.train import load\nfrom master9.control import demos\nmodel,_=load(OUTPUT)\nmodel=model.double()\ncontrol_report=demos(model,OUTPUT)\nfrom IPython.display import Image,display\ndisplay(Image(filename=str(OUTPUT/'control_results.png')))\n")
md('## استفاده با پارامتر و مسیر خود\nدر مثال زیر اعداد SI هستند. `from_re` ویسکوزیته را سازگار می‌سازد. برای مسیر دلخواه تابع `target(z)` در `master9/control.py` قابل تغییر است؛ z و x داخل کنترل بی‌بعدند. خروجی u جریان واقعی سیم‌پیچ نیست؛ مدل کالیبراسیون سخت‌افزار لازم است.')
code("from master9.physics import Parameters\np=Parameters.from_re(.8,mean=.015,rho=1000,x0=0,z0=-.08,vx0=.0005,vz0=.02)\nprint('Parameters:',p)\nprint('Re:',p.Re,'relaxation_s:',p.relaxation_s)\nprint('OUTPUTS:',OUTPUT)\nprint('ALL_PHYSICS_TESTS_PASSED:',physics_report['all_passed'])\nprint('ALL_CONTROL_TESTS_PASSED:',control_report['all_passed'])\n")
md('## مسیر CSV دلخواه — اختیاری\nعنوان فایل z_m,x_m است؛ همهٔ اعداد مترند. مسیر باید از نقطهٔ رهایش شروع شود، z صعودی داشته باشد و در دامنهٔ آموزش بماند. با فعال‌کردن RUN_MY_PATH می‌توانید نمونهٔ بسته را اجرا یا مسیر فایل خودتان را جایگزین کنید.')
code("RUN_MY_PATH=False\nif RUN_MY_PATH:\n    import numpy as np\n    from master9.control import follow\n    MY_PATH=ROOT/'examples/master9_target.csv'\n    data,report=follow(model,p,target_csv=MY_PATH)\n    np.savetxt(OUTPUT/'my_path_control.csv',data,delimiter=',',header='time_s,x_PINN_m,z_PINN_m,target_x_m,u,x_Radau_m,z_Radau_m,target_at_Radau_m',comments='')\n    (OUTPUT/'my_path_report.json').write_text(json.dumps(report,indent=2))\n    print(json.dumps(report,indent=2))\n    if not report['passed']:raise RuntimeError('Requested path did not pass independent acceptance.')\n")
n={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':4}
path=ROOT/'notebooks/master9.ipynb';path.write_text(json.dumps(n,ensure_ascii=False,indent=1));print(path)
