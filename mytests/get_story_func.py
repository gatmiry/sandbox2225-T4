
import requests
header = {'Authorization': '0NNFKdThRKtvSrCg5ApslcsVeg97r3bj'}
data = {'model': 'gpt-4o', 'prompt': 'Hi my name is Alice what about you?'}
url = 'https://blog.yintat.com/dqjoBCx0P2k7/api/models/unified-completion'
#response = requests.post(url, headers=header, json=data, timeout=600)
#print(response.text)

def get_stories(story):
    prompt1_text = 'write this story in a different style and change the names and characters: '+story
    prompt2_text = 'Now change this to a different story but use the same names and characters: '+story
    prompt1 = {'model': 'gpt-4o', 'prompt': prompt1_text}
    response1 = requests.post(url, headers=header, json=prompt1, timeout=600)
    import json
    data1 = json.loads(response1.text)
    story1 = data1['choices'][0]['text']
    ##
    prompt2 = {'model': 'gpt-4o', 'prompt': prompt2_text}
    response2 = requests.post(url, headers=header, json=prompt2, timeout=600)
    import json
    data2 = json.loads(response2.text)
    story2 = data2['choices'][0]['text']
    return {'story': story, 'story1': story1, 'story2':story2}

#story = ''' Once upon a time, there was a kind farmer. He had a big cow. The cow
#was sad. The farmer did not know why. One day, a little boy came to
#the farm. He saw the sad cow. The boy kneeled down to talk to the cow.
#"Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a
#friend." The kind farmer heard the cow. He wanted to help. So, he got
#another cow to be friends with the sad cow. The sad cow was happy now.
#They played together every day. And the kind farmer, the little boy,
#and the two cows all lived happily ever after.'''
#outputs = get_stories(story)
#print('first story is ', outputs['story1'])
#print('second story is ', outputs['story2'])


#import numpy as np
#turn_to_story1 = np.vectorize(lambda x: get_stories(story))
#response = turn_to_story1(np.array([story, story]))
#print('respone is ', response)