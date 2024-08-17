from calculate_embedding import EmbedCalculator
calculator = EmbedCalculator()

a = """Lily and Jake loved the swings. Every day after school, they would rush to the playground, eager to see who could swing the highest. "I’ll reach the sky first!" Lily called out as she pumped her legs, her red sneakers flying through the air.

Jake grinned, determined to catch up. "No way! I’m going to be the king of the sky!" His swing soared higher and higher, almost level with Lily’s.

They both laughed as they reached the peak of their swings, feeling like they were flying. When they finally slowed down, breathless from their race, they looked at each other with wide, happy eyes. "Let’s call it a tie!" Lily said, and Jake nodded, happy to share the victory.<|endoftext|>"""

b = """After their swing race, Lily and Jake headed to the giant slide, where their friends Mia and Ethan were waiting. The slide was the tallest in the playground, and sliding down it felt like a wild adventure.

"Let’s see who can slide down the fastest!" Mia announced, her curly hair bouncing with excitement.

Ethan quickly took his place at the top, and one by one, they each took a turn zooming down the shiny slide. The air rushed past them as they squealed in delight, trying to beat each other’s time.

At the end of the challenge, no one could agree on who was the fastest. "We’re all champions!" Ethan declared, and the four friends collapsed in a pile of giggles at the bottom of the slide, the joy of competition giving way to the warmth of friendship.<|endoftext|>"""

c = """Far away from the crowded playground, Leo and Ava discovered a hidden garden behind the big oak tree. This part of the park was quiet, overgrown with tall grass and colorful wildflowers, untouched by the busy laughter of other children.

"We’ve found a secret world," Ava whispered as they stepped carefully through the bushes, their sneakers brushing against the dew-kissed leaves.

Leo nodded, his eyes wide with wonder. "Let’s pretend we’re explorers in a jungle!" They began to weave through the flowers, imagining they were in a faraway land. They built a tiny fort out of sticks and leaves, pretending it was their base camp, and whispered about the imaginary creatures that lived in the trees.

The hours passed quietly in their secret garden until the sun began to set, casting long shadows across their hidden playground. "We’ll come back tomorrow," Ava said softly, and Leo agreed, knowing that their secret world would be waiting for them.<|endoftext|>"""


#######################################################################################################
aa = """ Once upon a time, there was a kind farmer. He had a big cow. The cow
was sad. The farmer did not know why. One day, a little boy came to
the farm. He saw the sad cow. The boy kneeled down to talk to the cow.
"Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a
friend." The kind farmer heard the cow. He wanted to help. So, he got
another cow to be friends with the sad cow. The sad cow was happy now.
They played together every day. And the kind farmer, the little boy,
and the two cows all lived happily ever after.<|endoftext|>"""


bb = """ In a small village, there was a gentle shepherd named Eli. He had a large, woolly sheep named Bella. But Bella was often quiet and seemed sad. Eli couldn't understand why.

One morning, a little girl named Mia visited the village. As she walked through the fields, she noticed Bella standing alone, looking downcast. Mia approached Bella and gently asked, "Why are you sad, sweet sheep?"

Bella sighed and softly replied, "I am lonely. I wish I had a friend."

Eli overheard their conversation and realized what Bella needed. Wanting to bring her joy, Eli decided to find another sheep to keep her company. Soon, Bella had a new friend named Clover.

From that day on, Bella and Clover were inseparable. They played together in the fields, happily grazing and enjoying the sunshine. And Eli, Mia, and the two sheep lived joyfully, knowing that friendship had brought happiness to Bella.<|endoftext|>"""



cc = """ Once upon a time, there was a kind farmer named Eli. He had a big cow named Bella. Bella was usually cheerful, but one day, she seemed very sad. Eli couldn’t figure out why.

One afternoon, a little girl named Mia came to visit the farm. As she wandered around, she noticed Bella standing by the barn, looking gloomy. Mia walked over and gently asked, "Why are you sad, Bella?"

Bella sighed and replied, "I lost my favorite bell. It used to hang around my neck, and now it’s gone. I miss the sweet sound it made."

Eli overheard this and wanted to help. He searched the barn, the fields, and even the stream, but he couldn’t find Bella’s bell. Determined to cheer her up, Eli decided to get a new bell—one that jingled just like her old one.

When Eli placed the new bell around Bella’s neck, her eyes sparkled with happiness. She shook her head, making the bell ring with a joyful sound. From that day on, Bella felt like herself again, happily strolling around the farm with her new bell.

Eli, Mia, and Bella all lived happily ever after, knowing that sometimes, a small gesture can make a big difference.<|endoftext|>"""

####################################################################################

aaa = '''Once upon a time there was a charming little girl called Lola. She loved to spin very quickly in circles, which made her laugh and giggle. Lola spun around the room so fast her toes flew up into the air!\n\nShe kept on spinning and spinning until her toes were all a jumble! She stopped and bent over to try to sort out her toes. But as soon as Lola tried to put them in the right places she started spinning again - just for fun.\n\nWhen Lola felt ready to slow down, she held out her arms and spun round and round until she came to a stop. She looked down and saw that her toes were all in the right places- a perfect pattern!\n\nLola kept spinning, happy in the knowledge that her toes looked charming and tidy. She spun and spun and spun, until she was too tired to spin any more. A happy smile filled her face and Lola knew that spinning was her favorite game.<|endoftext|>'''
bbb = '''Once upon a time, in a cozy little town, there was a delightful girl named Daisy. She adored twirling around until she felt like she was flying. With each spin, Daisy's laughter bubbled over, filling the room with joy.

As Daisy twirled, her feet lifted off the ground, whirling so fast that her toes seemed to blur into the air! She continued her dizzying dance, her feet intertwining in a playful tangle.

Eventually, Daisy paused and bent down to untangle her toes. Attempting to arrange them neatly, she burst into giggles and spun once more, swept up in the thrill of the swirl.

When Daisy finally felt ready to stop, she stretched out her arms wide and slowly spun to a gentle halt. Looking down, she was delighted to see her toes neatly aligned, forming a tidy pattern.

With a heart full of joy, Daisy resumed her twirling, reassured by the neat arrangement of her toes. She spun round and round, her energy unwinding until she could twirl no longer. Resting with a broad grin, Daisy realized that spinning was undoubtedly her favorite pastime.<|endoftext|>'''
ccc = '''Once upon a time, there was a delightful little girl named Lola who loved to dance. Every day, Lola would put on her favorite glittery shoes and dance around her room. Her laughter filled the air as she twirled and leapt, feeling as light as a feather.

One day, while dancing, Lola noticed something unusual about her shoes; they sparkled brighter than ever and seemed to move on their own! Curious and excited, Lola decided to go with the flow. She danced and spun, her shoes guiding her in graceful circles and elegant leaps.

As she danced, Lola found that the more she trusted her magical shoes, the more intricate her dance moves became. She spun around the room, and this time, her shoes created a dazzling trail of light that swirled around her.

Eventually, feeling a bit dizzy from all the excitement, Lola slowed her dance. As she came to a stop, she looked down and noticed that the light from her shoes had painted a beautiful pattern on the floor—a perfect starburst where she stood.

Thrilled by the magic of her dance, Lola continued to dance every day, eager to see what new patterns her enchanted shoes would create. She spun and danced until she was too tired to continue, always ending with a joyous smile. Dancing wasn't just Lola’s favorite game; it was her magical adventure.<|endoftext|>'''

#################################################################
a = '''Once upon a time, there was a hairy dog named Spot. Spot had a toy skull that he loved to play with. He would carry it around in his mouth and play fetch with it. One day, Spot's owner, a little girl named Lily, did not permit Spot to play with his skull. She was afraid that he might break it. Spot was sad and missed his favorite toy. Later, Lily saw how sad Spot was and decided to permit him to play with his skull again. Spot was so happy and wagged his tail. They played fetch together and had lots of fun. And from that day on, Spot and Lily played with the skull every day, and they were the best of friends.<|endoftext|>'''

b = '''In a bustling village, there lived a curious and mischievous squirrel named Squeaky. Squeaky had a favorite nut, a shiny acorn with intricate patterns on its shell. He loved carrying it around in his tiny paws and would often toss it in the air, catching it skillfully. This acorn was his prized possession, and he would play with it every chance he got.

One day, as Squeaky was about to start his usual game with the acorn, his friend, a wise old owl named Olive, fluttered down from her tree. Olive noticed Squeaky’s excitement but also his tendency to be a bit reckless with the acorn.

“Squeaky,” Olive hooted gently, “you should be careful with that acorn. It’s very special, and if you drop it too hard, it might crack.”

Squeaky looked up at Olive with wide eyes, realizing that she was right. He didn’t want to lose his favorite acorn. So, reluctantly, Squeaky decided to put the acorn away in a safe nook in his tree. But as the days passed, he missed playing with his beloved acorn. He tried playing with other nuts and berries, but none brought him the same joy.

Olive noticed how sad Squeaky had become and decided to visit him. “Squeaky, I see how much you miss your acorn,” she said. “I didn’t mean to make you give up something you love. Perhaps you can play with it again, just a little more carefully this time.”

Squeaky’s eyes lit up with excitement. “Really, Olive? You think it’s okay?”

Olive nodded, her wise eyes twinkling. “Of course, Squeaky. Just remember to be gentle.”

Filled with happiness, Squeaky rushed to retrieve his acorn. He held it carefully in his paws, appreciating its familiar weight and feel. Olive watched as Squeaky tossed the acorn into the air, this time with a little more care than before.

As they played together, Olive occasionally joined in, using her wings to send the acorn flying higher. Squeaky was thrilled, and they both enjoyed the game immensely.

From that day on, Squeaky played with his acorn every day, but he always remembered Olive’s advice. The game became even more special, knowing that he could enjoy his favorite toy while also keeping it safe.

Squeaky and Olive grew even closer, their bond strengthened by the shared understanding and fun they had together. And in the heart of the village, Squeaky’s joyful chirps and Olive’s gentle hoots filled the air as they played with the treasured acorn, their friendship growing stronger with each passing day.<|endoftext|>'''

c = '''In a quiet village surrounded by dense woods, there lived a curious squirrel named Squeaky. Squeaky had a special nut, a shiny acorn with intricate markings that made it stand out from all the others. He found it one day while foraging and immediately fell in love with it. Squeaky loved to roll it around, toss it in the air, and hide it in the soft moss by the trees.

One crisp autumn day, as Squeaky was playing with his acorn, his best friend, Olive the owl, noticed what he was doing. Olive, who was wise and had seen many things, was concerned.

“Squeaky,” Olive hooted softly, “that acorn is very unique. If you’re not careful, you might lose it or damage it. Maybe you should put it away for safekeeping.”

Squeaky, always trusting Olive’s wisdom, reluctantly agreed. He carefully tucked the acorn away in a hidden nook in his tree, deciding not to play with it anymore. But as the days passed, Squeaky found himself missing his favorite toy more and more. No other nut or berry could bring him the same joy.

Seeing how sad Squeaky had become, Olive started to feel guilty. She hadn’t meant to take away Squeaky’s happiness. One evening, as the sun set behind the trees, casting long shadows across the forest floor, Olive flew over to Squeaky’s tree.

“Squeaky,” Olive said, her voice gentle, “I can see that you miss playing with your acorn. I didn’t mean to make you so sad. Why don’t you take it out again? Just be careful with it.”

Squeaky’s eyes brightened immediately. “Really, Olive? You think it’ll be okay?”

Olive nodded. “Yes, Squeaky. It’s important to enjoy the things you love. Just remember to be gentle.”

Overjoyed, Squeaky scampered up to his nook and retrieved the acorn. As he held it in his paws, he felt a wave of happiness wash over him. He began to play with it again, but this time with a little more care.

Olive watched from a nearby branch, smiling as Squeaky’s joy returned. The two friends spent the evening playing together, with Olive occasionally swooping down to help Squeaky toss the acorn higher into the air.

From that day on, Squeaky continued to play with his acorn every day, but he always remembered Olive’s advice. The acorn remained in perfect condition, and Squeaky’s happiness was complete. Squeaky and Olive’s bond grew even stronger, and their shared moments of play became a cherished part of their days.

And so, in the quiet of the forest, Squeaky and Olive could often be seen playing together, the shiny acorn a symbol of their friendship and the joy of being able to enjoy life’s little pleasures.<|endoftext|>'''

embeddings = calculator.calculate_embedding([a,b,c])
print('embeddings are ', embeddings)

import torch
#val = a- b
print('close is ', torch.norm(embeddings[0] - embeddings[1], p=2), ' far is ', torch.norm(embeddings[0] - embeddings[2], p=2))