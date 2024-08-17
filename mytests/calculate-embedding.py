from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from mymodelnew4 import MyModel
checkpoint = 'nodotmodel3_weights_2024-08-05--20:10:40alaki'
#checkpoint = './testmodel3weights_2024-07-07--05:59:32'
#checkpoint = './model3weights_2024-07-04--16:34:15'
model = MyModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained('gpt2')


#raw_dataset = load_dataset(checkpoint)
#tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')
class EmbedCalculator():
    #def __init__(self, tokenizer):
    #    self.tokenizer = tokenizer
    def calculate_embedding(input_text):
        with torch.no_grad():
        #print('input text is ', input_text)
            outputs = tokenizer(input_text, return_tensors='pt')
            outputs['input_ids'] = outputs['input_ids'] + tokenizer.eos_token
        #print('im here')
        #print('outputs are ', outputs['input_ids'])

        #print('output ids are ', outputs['input_ids'])
        #print('len is ', len(outputs['input_ids'] - 1))
            hidden_embedding = model.encoder(outputs['input_ids']).last_hidden_state.detach()[0, outputs['input_ids'].shape[1]-1, :]
        return hidden_embedding

## a lot similar a & b! 
a = calculate_embedding("""In the charming village of Greendale, there was a boy named Noah who loved to roam the vast fields and woods that surrounded his home. One sunny morning, as he wandered further than he ever had before, he stumbled upon a hidden meadow filled with vibrant, glowing flowers and butterflies of every color. The air was filled with a sweet, enchanting fragrance that made the entire place feel magical.

In the middle of the meadow stood a magnificent oak tree with a small door at its base. Noah's curiosity got the better of him, and he gently pushed the door open. Inside, he found a secret world where animals could talk and plants could sing. He met a wise old fox named Finn, who offered to show him around.

Noah spent the day exploring this magical realm, making friends with the creatures and learning the secrets of the enchanted meadow. He discovered that the meadow was a sanctuary for all living things, a place of peace and harmony. As the sun began to set, Finn guided Noah back to the door, reminding him that he could return anytime he wanted to. Noah promised to come back and protect the meadow, ensuring its magic would never fade.""")
b = calculate_embedding("""In the tranquil town of Brooksville, a girl named Sophie had a passion for exploring the natural world around her. One warm afternoon, while hiking through the dense forest near her home, she came across a hidden path she had never seen before. Following the path, Sophie arrived at a secret glade bathed in golden sunlight and filled with blooming flowers and birds singing melodious tunes.

At the center of the glade stood a giant willow tree with a small, carved door at its base. Intrigued, Sophie opened the door and found herself in a magical world where animals spoke in gentle whispers and flowers danced in the breeze. She met an elderly rabbit named Hazel, who offered to be her guide.

Sophie spent the entire day exploring this enchanting glade, making new friends and uncovering the wonders of this hidden paradise. She learned that the glade was a haven for creatures of all kinds, a place where nature thrived in perfect harmony. As dusk approached, Hazel led Sophie back to the door, telling her she was welcome to return whenever she wished. Sophie vowed to come back often and help care for the glade, preserving its magic for future generations.""") 
c = calculate_embedding("""In a bustling city, a boy named Leo had a deep love for reading and history. One rainy afternoon, while visiting his grandparents' old mansion, he discovered a hidden staircase behind a tapestry in the grand hallway. The staircase led to a forgotten library filled with dusty books, ancient scrolls, and mysterious artifacts.

The library was dimly lit, with rays of sunlight filtering through the dusty windows. Leo spent hours poring over the books and manuscripts, uncovering tales of adventure, magic, and lost civilizations. Each book seemed to hold a new story, transporting him to different times and places.

Leo decided to document his discoveries, writing down the stories and knowledge he found in a journal. The forgotten library became his personal sanctuary, a place where he could delve into the mysteries of the past and fuel his imagination with tales of wonder and intrigue. The attic became a place of imagination and connection with her family's history, adding a personal and emotional depth to the story.""")

# a & b are just a little bit more similar!
a = calculate_embedding("""In the heart of a bustling city, there lay a serene park known for its towering trees and tranquil ponds. The park offered an oasis of greenery amidst the concrete jungle, where city dwellers sought respite from the hustle and bustle of urban life.

One sunny afternoon, Emily, a young professional with a love for nature, ventured into the park during her lunch break. She found a quiet bench beneath a sprawling oak tree, where she could escape the noise of traffic and enjoy the melody of birdsong.

As Emily relaxed in the park, she noticed a hidden pathway that led to a secluded garden filled with blooming flowers and fragrant herbs. Intrigued, she followed the pathway and discovered a small community garden tended by local volunteers. Inspired by their dedication, Emily decided to join the gardeners, cultivating plants and fostering a sense of community in the heart of the city.""")
b = calculate_embedding("""Not far from the city park, Daniel frequented a cozy café nestled on a bustling street corner. The café was a haven of warmth and aroma, where the scent of freshly brewed coffee mingled with the chatter of patrons.

One rainy morning, Daniel sought refuge from the storm inside the café, seeking solace in a quiet corner by the window. He found comfort in the familiar faces of regulars and the soothing ambiance of soft jazz playing in the background.

As Daniel sipped his coffee, he struck up a conversation with the café's owner, Maria, who shared her passion for creating a welcoming space for the community. Inspired by Maria's dedication, Daniel began volunteering at the café, organizing book readings and art exhibits that brought neighbors together and celebrated local talent.""") 
c = calculate_embedding("""Meanwhile, in a historic district of the city, Emma spent her weekends exploring the labyrinthine corridors of an old library that stood as a testament to bygone eras. The library was a treasure trove of knowledge, with shelves lined with dusty tomes and ancient manuscripts.

One quiet afternoon, Emma stumbled upon a forgotten section of the library that housed rare books and archives dating back centuries. She found herself captivated by tales of the city's rich history and the lives of its notable residents.

Inspired by the library's legacy, Emma joined a group of preservationists dedicated to restoring and preserving historical landmarks throughout the city. Together, they organized guided tours and educational workshops that highlighted the city's cultural heritage and fostered a sense of pride among its residents.""")


'''
# a little more similar a & b! (about art) infinity works better than l-2
a = calculate_embedding("""In a vibrant neighborhood known for its creative spirit, there stood an alley adorned with colorful street murals that transformed ordinary walls into canvases of expression. Each mural told a story, capturing the essence of the community and its diverse inhabitants.

One sunny afternoon, Maya, a budding artist with a passion for street art, decided to contribute to the alley's vibrant tapestry. Armed with cans of spray paint and a vision in mind, Maya set to work on a mural that depicted the neighborhood's rich cultural heritage and spirit of unity.

As Maya painted, she attracted the attention of Lucas, a fellow artist who admired her technique and storytelling ability. Impressed by Maya's dedication, Lucas offered to collaborate on a mural that celebrated their shared love for art and community. Together, they transformed a blank wall into a masterpiece that resonated with locals and visitors alike, inspiring conversations and fostering a sense of pride in their neighborhood's artistic identity.""")

b = calculate_embedding("""Not far from the street mural, Elena curated a gallery nestled in the heart of the city's art district. The gallery was a sanctuary of creativity, showcasing contemporary artworks that challenged perceptions and sparked dialogue among art enthusiasts.

One evening, during a much-anticipated gallery opening, Elena unveiled a collection of paintings by local artists that explored themes of identity and social justice. The artworks captivated the attention of attendees, who engaged in lively discussions about the power of art to provoke thought and evoke emotion.

Among the guests was Miguel, a young painter inspired by the gallery's commitment to showcasing diverse voices and perspectives. Impressed by Elena's curation, Miguel approached her with a proposal to collaborate on a series of workshops that would empower aspiring artists from underrepresented communities.""")

c = calculate_embedding("""Meanwhile, in a tranquil park on the outskirts of town, Marcus sculpted elegant figures from blocks of marble and granite. The sculpture garden was Marcus's sanctuary, where he found solace in the rhythmic chisel strokes that transformed stone into timeless works of art.

One afternoon, Emma, a local historian with a passion for storytelling, stumbled upon Marcus's sculpture garden during a leisurely stroll through the park. She was captivated by the grace and emotion conveyed in Marcus's sculptures, each one a testament to his craftsmanship and dedication.

Emma struck up a conversation with Marcus, eager to learn more about his creative process and the stories behind each sculpture. Inspired by their exchange, Emma proposed a collaboration to create an audio tour that would guide visitors through the sculpture garden, offering insights into Marcus's artistic journey and the historical context of his work.""")
'''


## a & b a little more similar (stories about egyptian temples) l-infinity works better than l-2 for our embedding
a = calculate_embedding("""Dr. Sarah Thompson, an esteemed archaeologist, had dedicated her life to uncovering the secrets of ancient Egypt. Her latest expedition led her to the sands of the Western Desert, where she had uncovered clues about a lost temple. After months of grueling excavation under the scorching sun, her team finally struck gold.

They unearthed a hidden entrance buried beneath the sands. Inside, the temple was remarkably well-preserved, with intricate hieroglyphs adorning the walls and statues of ancient gods standing guard. Among the artifacts was a beautifully carved altar, suggesting that this temple had been a significant place of worship. Sarah's discovery was hailed as a monumental find, providing new insights into the religious practices of ancient Egyptians.""")

b = calculate_embedding("""Professor James Carter, a renowned Egyptologist, had always been fascinated by the tales of a sacred sanctuary hidden in the Nile Delta. After years of research and piecing together ancient texts, he finally pinpointed its location. Leading a team of enthusiastic students, James embarked on an expedition to uncover the temple.

After weeks of excavation, they found the sanctuary's entrance concealed beneath dense vegetation. Inside, the temple was stunningly well-preserved, with vibrant frescoes depicting various gods and goddesses. In the heart of the temple stood a grand altar, indicating its importance as a center of worship. The rediscovery of the sanctuary shed new light on the religious life and artistic achievements of the ancient Egyptians, earning James and his team international acclaim.""")

c = calculate_embedding("""Dr. Emily Foster, an expert in ancient Egyptian architecture, was tasked with restoring a neglected shrine on the outskirts of Luxor. The shrine, dedicated to a lesser-known deity, had been forgotten over the centuries and was in a state of disrepair. Emily and her team began the painstaking work of restoration, carefully preserving the remaining artwork and structure.

As they worked, they uncovered hidden chambers within the shrine, each filled with statues and relics. The restoration process revealed the shrine's original beauty and significance, providing valuable insights into the daily worship practices and architectural styles of the time. Emily's dedication to preserving this piece of history was celebrated by scholars and history enthusiasts alike.""")

## three stories about ... where the first and second one are a bit more similar: 
a = calculate_embedding("""Ella, a talented singer-songwriter, started her career performing at small venues in her hometown. Her soulful voice and heartfelt lyrics resonated with audiences, attracting the attention of a local music producer. With his guidance, Ella recorded her first album, which garnered positive reviews and gained a loyal fan base.

As Ella's popularity grew, she embarked on a national tour, captivating audiences with her powerful performances. Along the way, she faced challenges such as stage fright and the pressures of fame, but her passion for music kept her grounded. Ella's journey culminated in winning a prestigious music award, solidifying her place as a respected artist in the industry.""")

b = calculate_embedding("""James, known for his versatile voice and songwriting skills, began his career singing cover songs in local bars. His talent caught the attention of established musicians, leading to collaborations on various projects. James thrived in the creative environment, learning from seasoned artists and contributing his own unique style to each collaboration.

Through these partnerships, James gained exposure to different genres and audiences, expanding his musical horizons. His ability to adapt and innovate in his songwriting earned him critical acclaim and a growing fan base. James' career flourished as he continued to collaborate with artists from around the world, creating memorable songs that resonated with listeners.""")

c = calculate_embedding("""Sarah, a gifted singer-songwriter, preferred a more introspective approach to her music career. She retreated to a secluded cabin in the mountains, where she found inspiration in nature and solitude. Sarah poured her emotions into heartfelt lyrics and melodies, crafting songs that spoke to personal struggles and triumphs.

Despite avoiding the spotlight, Sarah's music found its way to listeners through online platforms and word of mouth. Her intimate performances at small venues drew devoted fans who connected deeply with her authentic storytelling. Sarah's decision to prioritize artistic integrity over fame allowed her to maintain a meaningful career, inspiring others with her honest and raw songwriting.""")


val = a- b
print('close is ', torch.norm(a - b, p=20), ' far is ', torch.norm(a - c, p=20))