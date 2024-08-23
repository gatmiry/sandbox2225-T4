from calculate_embedding_T5 import EmbedCalculator
calculator = EmbedCalculator()

a = """Lily and Jake loved the swings. Every day after school, they would rush to the playground, eager to see who could swing the highest. "I’ll reach the sky first!" Lily called out as she pumped her legs, her red sneakers flying through the air.

Jake grinned, determined to catch up. "No way! I’m going to be the king of the sky!" His swing soared higher and higher, almost level with Lily’s.

They both laughed as they reached the peak of their swings, feeling like they were flying. When they finally slowed down, breathless from their race, they looked at each other with wide, happy eyes. "Let’s call it a tie!" Lily said, and Jake nodded, happy to share the victory."""

b = """After their swing race, Lily and Jake headed to the giant slide, where their friends Mia and Ethan were waiting. The slide was the tallest in the playground, and sliding down it felt like a wild adventure.

"Let’s see who can slide down the fastest!" Mia announced, her curly hair bouncing with excitement.

Ethan quickly took his place at the top, and one by one, they each took a turn zooming down the shiny slide. The air rushed past them as they squealed in delight, trying to beat each other’s time.

At the end of the challenge, no one could agree on who was the fastest. "We’re all champions!" Ethan declared, and the four friends collapsed in a pile of giggles at the bottom of the slide, the joy of competition giving way to the warmth of friendship."""

c = """Far away from the crowded playground, Leo and Ava discovered a hidden garden behind the big oak tree. This part of the park was quiet, overgrown with tall grass and colorful wildflowers, untouched by the busy laughter of other children.

"We’ve found a secret world," Ava whispered as they stepped carefully through the bushes, their sneakers brushing against the dew-kissed leaves.

Leo nodded, his eyes wide with wonder. "Let’s pretend we’re explorers in a jungle!" They began to weave through the flowers, imagining they were in a faraway land. They built a tiny fort out of sticks and leaves, pretending it was their base camp, and whispered about the imaginary creatures that lived in the trees.

The hours passed quietly in their secret garden until the sun began to set, casting long shadows across their hidden playground. "We’ll come back tomorrow," Ava said softly, and Leo agreed, knowing that their secret world would be waiting for them."""

import requests
url = 'https://blog.yintat.com/dqjoBCx0P2k7/api/models/unified-completion'
header =  {'Authorization': ''}
data = {'model': 'text-embedding-3-large', 'input': [a, b, c]}
response = requests.post(url, headers=header, json=data, timeout=600)
import json
data = json.loads(response.text)
print('data is ', )
import torch
#val = a- b
#print('close is ', torch.norm(embeddings[0] - embeddings[1], p=2), ' far is ', torch.norm(embeddings[0] - embeddings[2], p=2))

