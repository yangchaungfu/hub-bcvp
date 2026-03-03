text = """
“CPR!”“所有人离手！”“加除颤仪！”“再来一次！”李慕的意识在消失。他大约真的要死了。面对死亡，他并没有多少害怕，与其继续承受病痛的折磨，死，未必不是一种解脱。自父母去世以后，他在这个世界就了无牵挂。他早就做好了死的准备。医生焦急的叫喊声逐渐小了下去，李慕的意识沉入了无尽的深渊。祖洲。周国。北郡。荒山。乱葬岗前，几株杨树稀稀落落的站在那里，树叶被秋风吹的哗啦作响，乌鸦在高空盘旋数圈之后，落在凸起的坟丘上。某一刻，坟茔上觅食的乌鸦猛然惊起，煽动翅膀，飞上了天空。一阵沙沙的声音过后，两道身影从山道上走来。两人一高一矮，一胖一瘦，皆是身穿淡青色皂吏服，一路走到乱葬岗前，将一个破草席放下，其中的高个子长舒了口气，说道：“终于到了。”矮个子捕快看了看前方凸起的一片坟丘，忍不住哆嗦一下，说道：“赶紧挖吧，埋完了好回去，这可不是什么好地方，我总觉得背后凉飕飕的。”“大白天的，你怕什么？”高个子一屁股坐下，舒服的靠在一棵树上，说道：“累死了，先歇会再说，你说我们两个怎么就摊上这种倒霉差事。”“别这么说。”矮个子看着地上的尸体，悲从中来，说道：“李慕才倒霉，昨天还好好的，怎么就忽然……”高个子左右看了看，神秘道：“听说是被妖邪勾了魂。”“妖邪？”矮个子闻言一惊，“你听谁说的？”高个子吞了口唾沫，说道：“勾栏说书的不都这么说，生人的三魂七魄，对妖邪是大补之物，有些妖邪，专门勾人魂魄吞食修炼，连仵作都验不出来他的死因，不是妖邪作乱是什么。”风吹的树叶哗啦啦作响，再联想到恐怖的妖邪，矮个子只觉得周围阴风阵阵，连忙道：“别说了，赶快干活，干完了早点回去。”两人拿起铁锹，选了一块空地，开始挖坑。李慕没有亲人，又是穷光蛋一个，置办不起棺材，看在平日里一起共事的情分上，两人合伙出钱买了一张草席，料理他的后事，已经算是仁至义尽了。李慕睁开眼的时候，发现自己身在一处不知名的荒山，身旁，两个身穿古装制服的男人正在挖坑。他应该已经死了，就算没死的话，也应该在医院，这里是什么地方？这两个人在干什么？想要活埋他吗？即便是他已经没钱再支付医药费了，也不至于被活埋吧。他条件反射的坐起来。然后便是一怔。
"""
tokens = text.encode("utf-8") # raw bytes

# print('tokens1', tokens)

tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

# print('tokens2', tokens)


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


vocab_size = 276 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

# print('ids1', ids[:300])

merges = {}
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

print('merges', merges)
print('ids', ids[:300])

# 将 merges 和 ids 原样保存到文件中（Python 可直接 eval / literal_eval 读取）
with open("merges.txt", "w", encoding="utf-8") as f:
    f.write(repr(merges))

with open("ids.txt", "w", encoding="utf-8") as f:
    f.write(repr(ids))