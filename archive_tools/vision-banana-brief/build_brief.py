from pathlib import Path
from xml.sax.saxutils import escape
import json

from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT = Path('E:/Codex/2026-09-27/yo')
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)

for name, filename in [('Latin', 'arial.ttf'), ('LatinBold', 'arialbd.ttf'), ('Chinese', 'msyh.ttc'), ('ChineseBold', 'msyhbd.ttc')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / filename), subfontIndex=0))
pdfmetrics.registerFontFamily('Latin', normal='Latin', bold='LatinBold', italic='Latin', boldItalic='LatinBold')
pdfmetrics.registerFontFamily('Chinese', normal='Chinese', bold='ChineseBold', italic='Chinese', boldItalic='ChineseBold')

NAVY = colors.HexColor('#16324F')
TEAL = colors.HexColor('#087F8C')
TEXT = colors.HexColor('#243746')
MUTED = colors.HexColor('#5C6D7A')
LIGHT = colors.HexColor('#EDF5F7')
RULE = colors.HexColor('#D5E2E7')
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 96

REFS = [
    ('Vision Banana, paper v1', 'https://arxiv.org/html/2604.20329v1'),
    ('Vision Banana, official project demonstrations', 'https://vision-banana.github.io/'),
    ('Project method plan and completed V7 evidence', 'https://github.com/Maxnopnop/elec4240-marigold-depth/blob/main/METHOD_INNOVATION_PLAN.md'),
    ('GeoWizard, ECCV 2024', 'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03265.pdf'),
    ('Li et al., geometric constraints for laparoscopic depth, MICCAI 2025', 'https://papers.miccai.org/miccai-2025/paper/1214_paper.pdf'),
]

CONTENT = {
'en': {
 'title': 'Vision Banana: Evidence, Capabilities and a Feasible Research Contribution',
 'subtitle': 'ELEC4240 research briefing | Based on the linked paper v1 | 2 October 2026',
 'authors': 'HO, Chun Wai (21053878)  /  Tong, Man Hung (21064669)  /  MA, Shenhan (21041382)',
 'overview': '<b>Main assessment.</b> Vision Banana provides experimental evidence that an image generator can be adapted into a unified model for several perception tasks. For our project, the most practical research direction is to investigate when geometric consistency actually improves depth and normal prediction.',
 'h1': 'What does the paper establish?',
 'p1': 'The model instruction-tunes Nano Banana Pro using a mixture of its original data and added vision-task data. Different prompts select different tasks while sharing the same model weights. Its central design represents task outputs as RGB images that can be decoded into usable predictions. On the reported benchmarks, it matches or surpasses several specialist models and broadly retains image generation and editing capabilities. These results support the transferability of representations learned through generative pretraining. [1]',
 'pipeline': 'Input image + task instruction  >  Image generator  >  RGB output  >  Task-specific decoding',
 'h2': 'What tasks can it perform?',
 'tasks': [
   ['Task', 'Output and practical meaning'],
   ['Semantic segmentation', 'Assign a category, such as road, building or person, to each pixel.'],
   ['Instance segmentation', 'Separate individual objects that may belong to the same category.'],
   ['Referring-expression segmentation', 'Find an object described in language, such as a person in a pink shirt.'],
   ['Metric depth estimation', 'Predict per-pixel depth in metres from one image.'],
   ['Surface normal estimation', 'Predict the orientation of visible surfaces.'],
   ['Image generation and editing', 'Continue generating images and editing their content.'],
 ],
 'task_note': 'The project also visualizes point clouds obtained by combining predicted depth with camera intrinsics. This is a derived use of depth, rather than evidence of complete 3D reconstruction from any image. [2]',
 'boundary': '<b>Evidence boundary.</b> “Zero-shot transfer” still involves vision-task instruction tuning; the benchmark training splits are excluded from that tuning mixture. The model is not best on every dataset. Metric output does not guarantee exact measurement on arbitrary photos, and the results do not establish equivalent performance with a small model, little data or limited compute. [1]',
 'page2title': 'A research question grounded in our results',
 'position': 'Our Marigold implementation is an open-model study inspired by Vision Banana. Differences in backbone, data and training scale mean that it is neither a complete reproduction nor a direct test that could refute the original paper.',
 'question': '<b>Proposed question.</b> Under limited data and parameter-efficient adaptation, can the reliability of geometric supervision reduce negative transfer between metric depth and surface normal prediction?',
 'hresults': 'What our completed experiment shows',
 'results': [
   ['V7 setting', 'Depth AbsRel', 'Normal error'],
   ['Depth only', '0.37068', 'Not applicable'],
   ['Normals only', 'Not applicable', '39.83°'],
   ['Shared joint adaptation', '0.41819', '42.08°'],
   ['Joint adaptation + geometry', '0.39685', '45.07°'],
 ],
 'results_note': 'Lower is better. Values are averages across three seeds, using 128 training scenes, 32 previously observed validation scenes and 320 updates per run. Normal errors are measured against normals derived from depth, not independent normal ground truth. [3]',
 'interpret': 'Adding geometry reduced prediction-to-prediction angular disagreement from <b>48.54° to 31.11°</b>, while normal-target error rose from <b>42.08° to 45.07°</b>. The mean depth improvement over ordinary joint adaptation did not pass the exploratory Holm adjustment (p = 0.19344); normal degradation did (p = 0.00020). Thus, greater agreement can coexist with worse target accuracy. [3]',
 'cause': 'This motivates an investigation, but does not isolate its cause. Label sensitivity, short training, shared adapter capacity and optimization remain competing explanations.',
 'hmethod': 'Candidate modification: reliability-weighted geometry',
 'steps': [
   ('Estimate label stability.', 'Derive normal labels from each training depth map using several fixed neighbourhood sizes or smoothing scales. Measure angular disagreement at each valid location.'),
   ('Weight the geometric term.', 'Assign higher weights to stable locations and lower weights to unstable ones. Retain both original supervised objectives and the existing validity mask.'),
   ('Validate the proxy first.', 'Use known planes, corners, controlled noise and missing depth to test whether instability predicts actual normal error. Stable labels can still be wrong.'),
   ('Keep inference unchanged.', 'Compute weights from training labels and detach them from optimization. Ground-truth depth and reliability maps are not required at inference.'),
 ],
 'status': '<b>Status:</b> this candidate weighting method has not been implemented, trained or evaluated. No improvement is claimed.',
 'page3title': 'How to establish a credible contribution',
 'hablation': 'Minimum informative ablation',
 'ablations': [
   ['Condition', 'Question it answers'],
   ['Joint model without geometry', 'What is the shared-model baseline?'],
   ['Uniform geometric penalty', 'What does the existing constraint change?'],
   ['Weaker uniform penalty', 'Is a gain explained simply by less regularization?'],
   ['Reliability-weighted penalty', 'Does the proposed weighting rule help?'],
   ['Spatially shuffled weights', 'Does selecting the correct locations matter?'],
 ],
 'controls': 'Retain matching depth-only and normal-only controls. Match training data, initialization, exposure, resolution and checkpoint selection. Report all seeds, both task errors and compute cost. Keep the evaluation mask common to every method.',
 'validation': 'Use current scenes for development. Freeze the selected method and statistical comparisons before testing genuinely unobserved scenes. To claim no meaningful normal degradation, predefine an acceptable margin and evaluate the corresponding confidence bound; a non-significant difference alone is insufficient.',
 'hnovel': 'Where the novelty must be precise',
 'novel': 'Joint depth-normal diffusion is already studied by GeoWizard [4]. Uncertainty-weighted geometric consistency also has precedents: Li et al. use local point-to-plane residuals to reduce depth-to-normal conversion bias [5]. Therefore, joint prediction, geometric consistency and uncertainty weighting are not new in themselves.',
 'contribution': '<b>Potential contribution:</b> a specific spatial weighting rule based on the sensitivity of derived labels to their construction scale, together with controlled evidence of when it helps or fails during low-cost generative-model adaptation. This is a candidate course-project contribution; publication-level novelty needs a broader comparison with prior work.',
 'next': '<b>Recommended sequence:</b> stabilize single-task training, validate the reliability proxy, run the controlled ablations, and then evaluate the frozen method independently.',
 'titlelabel': 'Suggested working title',
 'workingtitle': 'Reliability-Aware Geometric Supervision for Parameter-Efficient Depth and Normal Adaptation',
 'refsheading': 'Sources',
 'footer': 'ELEC4240 | Vision Banana research briefing',
},
'zh': {
 'title': 'Vision Banana：论文证据、任务能力与可行创新方向',
 'subtitle': 'ELEC4240 研究分析 | 基于最初提供的论文 v1 | 2026 年 10 月 2 日',
 'authors': 'HO, Chun Wai (21053878)  /  Tong, Man Hung (21064669)  /  MA, Shenhan (21041382)',
 'overview': '<b>核心判断。</b>Vision Banana 为“图像生成模型可以适配成支持多种感知任务的统一模型”提供了实验证据。结合本项目，最适合研究的方向是：几何一致性在什么情况下才能真正改善深度与法线预测。',
 'h1': '论文本身说明了什么？',
 'p1': '作者将新增视觉任务数据混入原有数据，对 Nano Banana Pro 进行指令微调。同一套模型权重通过不同提示执行不同任务，核心设计是把任务输出表示成可以解码的 RGB 图片。在报告的评测中，它达到或超过若干专用模型，同时基本保留图像生成和编辑能力。这支持“生成预训练学到了可迁移视觉表示”的判断。[1]',
 'pipeline': '输入照片 + 任务指令  >  图像生成模型  >  RGB 结果图  >  任务解码',
 'h2': '可以完成哪些任务？',
 'tasks': [
   ['任务', '输出及实际含义'],
   ['语义分割', '判断每个像素属于道路、建筑、行人等哪一类。'],
   ['实例分割', '区分同一类别中的不同物体，例如不同的人。'],
   ['指代表达分割', '根据“穿粉色衣服的人”等语言描述找到目标。'],
   ['米制深度估计', '从单张照片预测以米为单位的逐像素深度。'],
   ['表面法线估计', '预测可见物体表面的朝向。'],
   ['图像生成与编辑', '继续生成图片和修改图片内容。'],
 ],
 'task_note': '项目还展示了将预测深度与相机内参结合得到的点云。这是深度的下游用途，不能据此认定它能从任意照片完整重建三维场景。[2]',
 'boundary': '<b>证据边界。</b>论文中的“零样本迁移”仍经过视觉任务指令微调，只是微调混合数据不包含相应评测数据集的训练集。模型并非在每个数据集上都领先。米制输出不保证任意照片都能精确测距，结果也不能直接推广到小模型、小数据和有限算力条件。[1]',
 'page2title': '从现有结果出发确定研究问题',
 'position': '我们的 Marigold 实现是受 Vision Banana 启发的开放模型研究。底座、数据和训练规模存在差异，因此当前工作既不是完整复现，也不能直接反驳原论文。',
 'question': '<b>建议研究问题。</b>在有限数据和参数高效适配条件下，能否根据几何监督的可靠性，减少米制深度与表面法线联合训练中的负迁移？',
 'hresults': '已完成实验提供了什么依据？',
 'results': [
   ['V7 设置', '深度 AbsRel', '法线误差'],
   ['仅训练深度', '0.37068', '不适用'],
   ['仅训练法线', '不适用', '39.83°'],
   ['共享参数联合训练', '0.41819', '42.08°'],
   ['联合训练 + 几何约束', '0.39685', '45.07°'],
 ],
 'results_note': '数值越低越好。表中为三个种子的平均结果：128 个训练场景、32 个此前已观察的验证场景，每次训练 320 次更新。法线误差相对于深度派生标签计算，而非独立法线真值。[3]',
 'interpret': '加入几何约束后，两个预测之间的角度不一致从 <b>48.54° 降至 31.11°</b>，但法线相对标签的误差从 <b>42.08° 升至 45.07°</b>。相对普通联合训练，深度平均改善未通过探索性 Holm 校正（p = 0.19344），法线退化通过了该检验（p = 0.00020）。因此，预测更一致仍可能伴随目标误差增大。[3]',
 'cause': '这构成研究动机，但尚未确定原因。标签敏感性、训练不足、共享适配器容量和优化过程仍是需要区分的解释。',
 'hmethod': '候选改进：根据可靠性加权几何约束',
 'steps': [
   ('估计标签稳定性。', '对同一训练深度，用多个固定邻域或平滑尺度构造法线，在有效位置测量法线方向的差异。'),
   ('有选择地施加约束。', '稳定位置赋较高权重，不稳定位置赋较低权重；保留两个原有监督目标和现有有效性掩码。'),
   ('先验证可靠性指标。', '用已知真值的平面、折角、受控噪声和深度缺失，检验不稳定程度是否能预测真实法线误差。稳定的标签也可能是错的。'),
   ('保持推理流程不变。', '权重从训练标签预先计算，并阻断其优化梯度。推理时无需真实深度或可靠性图。'),
 ],
 'status': '<b>当前状态：</b>这一候选加权方法尚未实现、训练或评估，目前不声称它已经带来改进。',
 'page3title': '如何形成可信的研究贡献',
 'hablation': '最小且有解释力的消融实验',
 'ablations': [
   ['设置', '要回答的问题'],
   ['无几何约束的联合模型', '共享模型本身表现如何？'],
   ['均匀几何约束', '现有几何约束带来什么变化？'],
   ['较弱的均匀约束', '改善是否仅来自减少正则化强度？'],
   ['可靠性加权约束', '提出的空间加权规则是否有效？'],
   ['空间位置打乱的权重', '是否真的选对了施加约束的位置？'],
 ],
 'controls': '同时保留匹配的深度单任务和法线单任务对照。各组匹配训练数据、初始化、图像曝光次数、处理分辨率和检查点选择规则；报告全部种子、两个任务误差及计算成本。所有方法使用相同评估掩码。',
 'validation': '现有场景用于开发。在真正未观察过的场景上测试前，冻结选定方法和统计比较方案。若要声称“法线没有实质退化”，需预先定义可接受的退化幅度，并检查相应置信界；差异不显著本身不能证明无损害。',
 'hnovel': '创新点必须具体到哪里？',
 'novel': 'GeoWizard 已研究扩散模型联合预测深度与法线 [4]。不确定性加权几何一致性也已有先例：Li 等人用局部点到平面的残差，减轻深度转换为法线时的偏差 [5]。因此，联合预测、几何一致性和不确定性加权这些方向本身均不是全新概念。',
 'contribution': '<b>可以争取的贡献：</b>提出一种基于“派生标签对构造尺度的敏感性”的具体空间加权规则，并通过受控实验说明它在低成本生成式模型适配中何时有效、何时失效。这是候选课程项目贡献；论文发表级新颖性仍需要更广泛的相关工作比较。',
 'next': '<b>建议顺序：</b>先稳定单任务训练，再验证可靠性指标，随后完成受控消融，最后独立评估冻结的方法。',
 'titlelabel': '建议英文题目',
 'workingtitle': 'Reliability-Aware Geometric Supervision for Parameter-Efficient Depth and Normal Adaptation',
 'refsheading': '参考来源',
 'footer': 'ELEC4240 | Vision Banana 研究分析',
},
}

CONTENT['en'].update({
 'plain_title': 'The proposed idea in plain language',
 'plain_summary': '<b>Everyday version:</b> let reliable regions contribute more when checking whether depth and surface orientation agree. In uncertain regions, apply less pressure to force agreement.',
 'plain_terms': [
   ['Term', 'Meaning in this project'],
   ['Label', 'A target surface normal computed from training depth: the direction in which the local surface faces.'],
   ['Construction scale', 'The neighbourhood or smoothing scale used to compute that normal. It is not the physical size of an object.'],
   ['Sensitivity', 'How much the computed direction changes when the construction scale changes.'],
   ['Spatial weighting', 'Use a different strength of geometric supervision at each image location.'],
 ],
 'plain_example_heading': 'Example: a wall next to a table edge',
 'plain_example': 'On a flat wall, small and large neighbourhoods often produce similar normal directions. At a table edge, a small neighbourhood may cover only the tabletop, while a larger one includes the background. The computed direction can change substantially. Depth noise and missing measurements can also create instability.',
 'plain_rule': 'The proposed rule measures this change, gives stable locations larger weights, and gives unstable locations smaller weights. These weights affect only the additional loss that encourages predicted depth and predicted normals to agree. Both ordinary supervised losses remain in use. The model still predicts the entire image.',
 'plain_failure': '<b>Why validation matters:</b> a real curved surface or sharp edge can also change with scale. Conversely, a biased depth map can produce stable but incorrect normals. The rule must distinguish unreliable supervision from useful fine structure; it cannot assume that stability equals correctness.',
 'plain_test': '<b>What “verify” means:</b> first test whether the score tracks true normal error on controlled geometry. Then compare matched trained models, including weaker uniform and shuffled-weight controls, on full-mask depth and normal accuracy. Lower consistency loss alone is not success.',
 'plain_app_heading': 'Potential applications and limits',
 'plain_apps': [
   ['Setting', 'Potential value if the method is validated'],
   ['Indoor scene reconstruction', 'Reduce geometric artifacts caused by unreliable training labels in walls, floors and object boundaries.'],
   ['AR and virtual object placement', 'Provide depth and surface orientation for placing virtual objects and reasoning about occlusion.'],
   ['Low-budget RGB-D adaptation', 'Study whether a small labelled dataset can adapt an open model more reliably with limited GPU memory.'],
 ],
 'plain_limit': 'These are potential uses, not demonstrated downstream improvements. The current model is not validated for safety-critical navigation. Normal consistency alone cannot determine absolute distance; metric-depth supervision remains necessary.',
 'plain_pitch': '<b>Presentation sentence:</b> We propose to trust stable geometric supervision more, and test whether doing so helps a small adapted model predict both depth and surface orientation more reliably.',
})
CONTENT['zh'].update({
 'plain_title': '通俗解释：这项方法到底在做什么？',
 'plain_summary': '<b>一句话表达：</b>在可靠的地方，让深度和表面朝向互相检查；在不确定的地方，少强迫它们一致。',
 'plain_terms': [
   ['术语', '在本项目中的具体含义'],
   ['标签', '从训练深度计算出的目标法线，也就是局部表面朝向。'],
   ['构造尺度', '计算法线时使用的邻域或平滑尺度，不是物体的实际大小。'],
   ['敏感性', '改变构造尺度后，计算得到的法线方向变化有多大。'],
   ['空间加权', '图像中不同位置使用不同强度的几何监督。'],
 ],
 'plain_example_heading': '例子：墙面与桌子边缘',
 'plain_example': '在平整墙面上，小邻域和大邻域通常会算出相近的法线方向。到了桌子边缘，小邻域可能只包含桌面，大邻域却包含背景，计算出的方向就可能明显变化。深度噪声和缺失测量也可能造成这种不稳定。',
 'plain_rule': '候选规则先测量这种变化，再给稳定位置较大的权重，给不稳定位置较小的权重。权重只作用于“让预测深度和预测法线彼此一致”的额外损失；两个普通监督损失继续保留，模型仍然预测完整图像。',
 'plain_failure': '<b>为什么必须验证：</b>真实曲面和锐利边缘也可能随尺度变化；有系统偏差的深度图也可能产生稳定但错误的法线。规则需要区分“不可靠监督”和“有用细节”，不能直接把稳定等同于正确。',
 'plain_test': '<b>“验证”的含义：</b>先用已知真值的受控几何，检查分数是否对应真实法线误差；再与较弱均匀约束、打乱权重等匹配训练模型比较，检查完整有效区域的深度和法线精度。仅仅一致性损失下降不算成功。',
 'plain_app_heading': '潜在应用场景与边界',
 'plain_apps': [
   ['场景', '如果方法验证有效，可能带来的用途'],
   ['室内场景重建', '减少不可靠训练标签在墙面、地面和物体边界引入的几何伪影。'],
   ['AR 与虚拟物体摆放', '提供深度和表面朝向，用于虚拟物体放置和遮挡关系判断。'],
   ['低预算 RGB-D 模型适配', '研究少量标注数据能否在有限显存下，更可靠地适配开放模型。'],
 ],
 'plain_limit': '以上属于潜在用途，尚未验证下游改善。当前模型未达到安全关键导航的验证要求。单靠法线一致性不能确定绝对距离，仍需要米制深度监督。',
 'plain_pitch': '<b>汇报时可以这样说：</b>我们希望让模型在可靠的地方多做几何互相校验，在不可靠的地方少受干扰，再通过实验检验它是否能同时提高深度和表面朝向预测的可靠性。',
})

def make_styles(lang):
    font, bold = ('Chinese', 'ChineseBold') if lang == 'zh' else ('Latin', 'LatinBold')
    body_size = 10.2 if lang == 'zh' else 10.3
    leading = 15.8 if lang == 'zh' else 14.3
    base = dict(fontName=font, textColor=TEXT, fontSize=body_size, leading=leading, spaceAfter=7, wordWrap='CJK' if lang == 'zh' else None)
    styles = {'body': ParagraphStyle('body', **base)}
    for k, extras in {
      'title': dict(fontName=bold, fontSize=21 if lang == 'en' else 20, leading=25.5, textColor=NAVY, spaceAfter=10),
      'subtitle': dict(fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=5),
      'authors': dict(fontName='Latin', fontSize=8, leading=11, textColor=MUTED, spaceAfter=12),
      'section': dict(fontName=bold, fontSize=12.2, leading=16.2, textColor=NAVY, spaceBefore=8, spaceAfter=7, keepWithNext=True),
      'page_title': dict(fontName=bold, fontSize=19, leading=24, textColor=NAVY, spaceAfter=12),
      'small': dict(fontSize=8.6, leading=12, textColor=MUTED, spaceAfter=7),
      'cell': dict(fontSize=9.1 if lang == 'en' else 9.5, leading=12.5 if lang == 'en' else 14.2, spaceAfter=0),
      'header': dict(fontName=bold, fontSize=9.3, leading=12.5, textColor=colors.white, spaceAfter=0),
      'ref': dict(fontName='Latin', fontSize=8.2, leading=11.4, spaceAfter=4),
      'callout': dict(fontSize=10.3, leading=15, spaceAfter=0),
      'pipeline': dict(fontSize=9 if lang == 'en' else 10, leading=14, textColor=TEAL, spaceAfter=8),
    }.items():
        args = dict(base)
        args.update(extras)
        styles[k] = ParagraphStyle(k, **args)
    return styles

def build(lang):
    data = CONTENT[lang]
    styles = make_styles(lang)
    story = []
    def p(text, style='body'):
        return Paragraph(text, styles[style])
    def add(key, style='body'):
        story.append(p(data[key], style))
    def box(key):
        t = Table([[p(data[key], 'callout')]], colWidths=[WIDTH])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),0.5,RULE),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
        story.extend([t, Spacer(1, 10)])
    def table(key, widths):
        cells = [[p(escape(cell), 'header' if i == 0 else 'cell') for cell in row] for i, row in enumerate(data[key])]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F3F7F9'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.7,NAVY),('LINEBELOW',(0,1),(-1,-1),0.35,RULE),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t, Spacer(1, 8)])

    add('title','title'); add('subtitle','subtitle'); add('authors','authors'); box('overview')
    add('h1','section'); add('p1'); add('pipeline','pipeline')
    add('h2','section'); table('tasks',[157,WIDTH-157]); add('task_note','small'); add('boundary')
    story.append(PageBreak())
    add('page2title','page_title'); add('position'); box('question')
    add('hresults','section'); table('results',[WIDTH-210,105,105]); add('results_note','small'); add('interpret'); add('cause')
    add('hmethod','section')
    for i, (label, desc) in enumerate(data['steps'], 1):
        story.append(p(f'<b>{i}. {label}</b> {desc}'))
    add('status','small')
    story.append(PageBreak())
    add('page3title','page_title'); add('hablation','section'); table('ablations',[190,WIDTH-190]); add('controls'); add('validation')
    add('hnovel','section'); add('novel'); add('contribution'); add('next')
    story.append(p('<b>' + data['titlelabel'] + ':</b> ' + data['workingtitle']))
    add('refsheading','section')
    for i,(title,url) in enumerate(REFS,1):
        story.append(p(f'[{i}] <link href="{url}" color="#087F8C">{escape(title)}</link>', 'ref'))


    filename = OUT / ('Vision_Banana_Research_Analysis_EN.pdf' if lang == 'en' else 'Vision_Banana_Research_Analysis_ZH.pdf')
    def decorate(c, doc):
        c.saveState()
        c.setFillColor(TEAL)
        c.rect(48, PAGE_H-25, 37, 3, fill=1, stroke=0)
        c.setStrokeColor(RULE); c.setLineWidth(.5); c.line(48,38,PAGE_W-48,38)
        c.setFont('Chinese' if lang == 'zh' else 'Latin',7.5); c.setFillColor(MUTED)
        c.drawString(48,25,data['footer'])
        c.setFont('Latin',7.5); c.drawRightString(PAGE_W-48,25,f'{doc.page} / 3')
        c.restoreState()
    doc = SimpleDocTemplate(str(filename), pagesize=A4, rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=50,
        title=data['title'],author='HO, Chun Wai; Tong, Man Hung; MA, Shenhan',subject='Vision Banana paper analysis and a proposed ELEC4240 research direction',
        pageCompression=1)
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    return filename

if __name__ == '__main__':
    outputs = [build('zh'),build('en')]
    checks = []
    for filename in outputs:
        reader = PdfReader(filename)
        all_text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        links = []
        for page in reader.pages:
            for annotation in page.get('/Annots',[]):
                action = annotation.get_object().get('/A',{})
                if '/URI' in action:
                    links.append(action['/URI'])
        assert len(reader.pages)==3, (filename,len(reader.pages))
        assert all(x in all_text for x in ['48.54','31.11','42.08','45.07','0.19344','0.00020','21053878','21064669','21041382'])
        assert set(url for _,url in REFS).issubset(set(links)), links
        assert '\ufffd' not in all_text
        checks.append({'path':str(filename),'pages':len(reader.pages),'bytes':filename.stat().st_size,'links':len(links),'characters':len(all_text)})
    (Path(__file__).parent/'pdf_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))
