from openai import OpenAI
client = OpenAI()


class SanguoAugmenter():
    def __init__(self):
        self.biography = {}
        self.client = OpenAI()
    
    def index_doc(self): 
        with open('data/sanguo/baihua2.txt') as fw:
            lines = fw.readlines()
        
        with open('data/subtitles.txt') as ft:
            subtitles_raw = ft.readlines()
        subtitles = []
        for subtitle in  subtitles_raw:
            subtitles.append(subtitle.strip())
        
        subidx = 0 
        subtitle_nos = {}
        cur = None 
        for line_no, line in enumerate(lines):
            if subidx >= len(subtitles):
                break
            if subtitles[subidx] in line:
                subtitle_nos[subtitles[subidx]] = line_no
                if subidx >= len(subtitles):
                    break
                if '三国志' not in subtitles[subidx] :
                    cur = subtitles[subidx].strip()
                    if cur[0] == '（':
                        cur = cur[1:-1]
                if '三国志' not in subtitles[subidx]:
                    self.biography[cur] = []
                    self.biography[cur].append(line)
                subidx = subidx + 1
            elif cur is not None:
                self.biography[cur].append(line)
            else:
                pass
        
    def get_bio(self, name):
        names = self.biography.keys()
        story = None
        for cname in names:
            if name in cname and '夫人' not in cname:
                story = ''.join(self.biography[cname])
        return story
                
        
    def key_person_identify(self,question):
        response = client.chat.completions.create(
            model="gpt-4",
            messages= [{"role":"system","content":"你用于识别历史问题中关键人物，从而进行相关资料的查询,只有当一个人是问题主要人物时回答一个人物，当涉及到两个同等程度的人物时回答无，例如:诸葛亮是否杀了马谡，回答：诸葛亮，例如魏国是否比蜀国强，回答无，例如马谡是否丢失街亭，回答马谡，例如徐晃是否比张辽强回答无,例如刘备是否和孙权联盟回答无"}, {'role':'user','content':question}])
        return response

    def query(self, question):           
        res = self.key_person_identify(question)
        name = res.choices[0].message.content
        if name == '无':
            return None
        bio = self.get_bio(name) 
        if bio == None:
            return None

        response = client.chat.completions.create(
            model="gpt-4",
            messages= [{"role":"system","content":"你用于根据史料回答历史问题，你需要非常小心的阅读资料，只根据资料来回答，当无法回答时只需打出四个字[无法回答], 如果可以回答请回答并在[]中引用原文来支持你的回答 "}, {'role':'user','content':f'问题:{question}\n相关资料{bio[:4000]}'}])
        res_str = response.choices[0].message.content

        if '无法' in res_str:
            res_str = None
         
        return res_str

c = SanguoAugmenter()
c.index_doc()
res= c.get_bio('孙权')
print(res)
