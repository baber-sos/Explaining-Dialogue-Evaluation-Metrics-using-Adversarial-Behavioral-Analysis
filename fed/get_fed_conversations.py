import json

annotated_responses = json.load(open('fed_data.json'))

conversation_count = 0
count_by_agent = dict()
for conversation in annotated_responses:
    if 'response' not in conversation:
        conversation_count += 1
        count_by_agent.setdefault(conversation['system'], 0)
        count_by_agent[conversation['system']] += 1
        print(f'Conversation Number: {conversation_count}')
        print(conversation['context'])
        print(f'System which undertook the conversation: {conversation["system"]}')
        print('---------------')
print('Total Conversations:', conversation_count)
print('Conversation Count by Agent:', count_by_agent)