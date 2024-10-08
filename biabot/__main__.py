# assignment/__main__.py

import click
from collections import defaultdict
import os
import pandas as pd
import sys

from .chat import ChatManager
from .config import ChatConfig, ENV_CONFIG
from .io import write_jsonl, read_jsonl
from .llm import OpenAIProvider, ChatConfig
from .models.conversation import ConversationModel
from .persona import Persona

@click.group()
@click.option('--lancedb-uri', default=ENV_CONFIG['lancedb_uri'], help='URI for LanceDB')
@click.pass_context
def cli(ctx, lancedb_uri):
    ctx.obj = ConversationModel.from_uri(
        lancedb_uri=lancedb_uri,
        device='cpu'
    )

@cli.command()
@click.pass_obj
def list_conversations(cvm: ConversationModel):
    """List all conversations"""
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 100)
    df: pd.DataFrame = cvm.collection.to_pandas()
    conversations = df.groupby(['document_type', 'user_id', 'persona_id', 'conversation_id']).size().reset_index(name='messages')
    click.echo(conversations)

@cli.command()
@click.pass_obj
def matrix(cvm: ConversationModel):
    """List all conversations"""
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 100)
    df: pd.DataFrame = cvm.get_conversation_report()
    df.columns = [s[:2] for s in df.columns]
    click.echo(df)

@cli.command()
@click.argument('user_id')
@click.argument('persona_id')
@click.argument('conversation_id')
@click.pass_obj
def display_conversation(cvm: ConversationModel, user_id, persona_id, conversation_id):
    """Display a specific conversation"""
    history = cvm.get_conversation_history(user_id, persona_id, conversation_id)
    for _, row in history.iterrows():
        click.echo(f"{row['role']}: {row['content']}\n")

@cli.command()
@click.argument('user_id')
@click.argument('persona_id')
@click.argument('conversation_id')
@click.pass_obj
def delete_conversation(cvm: ConversationModel, user_id, persona_id, conversation_id):
    """Delete a specific conversation"""
    cvm.collection.delete(f"user_id = '{user_id}' and persona_id = '{persona_id}' and conversation_id = '{conversation_id}'")
    click.echo(f"Conversation {conversation_id} for user {user_id} with {persona_id} has been deleted.")

@cli.command()
@click.argument('user_id')
@click.argument('persona_id')
@click.argument('conversation_id')
@click.option('--to-user-id', default=None, help='User ID for whom to apply the conversation')
@click.option('--to-persona-id', default=None, help='Persona ID for whom to apply the conversation')
@click.option('--to-conversation-id', default=None, help='Conversation ID to use for the conversation')
@click.pass_obj
def copy_conversation(cvm: ConversationModel, user_id, persona_id, conversation_id, to_conversation_id, to_user_id, to_persona_id):
    """Copy a conversation to a new conversation ID"""
    if to_conversation_id is None and to_user_id is None and to_persona_id is None:
        click.echo("Please provide at least one of to_conversation_id, to_user_id, or to_persona_id.")
        return

    if user_id == to_user_id and persona_id == to_persona_id and conversation_id == to_conversation_id:
        click.echo("Cannot copy a conversation to itself.")
        return
    
    source_history = cvm.get_conversation_history(user_id, persona_id, conversation_id)
    
    if to_user_id is None:
        to_user_id = user_id
    if to_persona_id is None:
        to_persona_id = persona_id
    if to_conversation_id is None:
        to_conversation_id = conversation_id
    
    for _, row in source_history.iterrows():
        cvm.insert(**row.to_dict(), conversation_id=to_conversation_id, user_id=to_user_id, persona_id=to_persona_id)
    
    click.echo(f"Conversation {conversation_id} for user {user_id} with {persona_id} has been copied to {to_user_id} with {to_persona_id} as {to_conversation_id}.")

@cli.command()
@click.argument('user_id')
@click.argument('persona_id')
@click.argument('conversation_id')
@click.option('--to-user-id', default=None, help='User ID for whom to apply the conversation')
@click.option('--to-persona-id', default=None, help='Persona ID for whom to apply the conversation')
@click.option('--to-conversation-id', default=None, help='Conversation ID to use for the conversation')
@click.pass_obj
def rename_conversation(cvm: ConversationModel, user_id, persona_id, conversation_id, to_conversation_id, to_user_id, to_persona_id):
    """Rename a conversation to a new conversation ID"""
    if to_conversation_id is None and to_user_id is None and to_persona_id is None:
        click.echo("Please provide at least one of to_conversation_id, to_user_id, or to_persona_id.")
        return

    if user_id == to_user_id and persona_id == to_persona_id and conversation_id == to_conversation_id:
        click.echo("Cannot rename to the same conversation.")
        return
    
    if to_user_id is None:
        to_user_id = user_id
    if to_persona_id is None:
        to_persona_id = persona_id
    if to_conversation_id is None:
        to_conversation_id = conversation_id

    history = cvm.get_conversation_history(user_id, persona_id, conversation_id)
    if len(history) == 0:
        click.echo("No conversation found.")
        return

    cvm.collection.update(f"user_id = '{user_id}' and persona_id = '{persona_id}' and conversation_id = '{conversation_id}'",
                          {"conversation_id": to_conversation_id, "user_id": to_user_id, "persona_id": to_persona_id})

    click.echo(f"Conversation {conversation_id} with user {user_id} with {persona_id} has been renamed to {to_conversation_id} for user {to_user_id} with {to_persona_id}.")

@cli.command()
@click.option('--workdir_folder', default=ENV_CONFIG['workdir_folder'], help='working directory')
@click.option('--filename', default=None, help='output file')
@click.argument('user_id')
@click.argument('persona_id')
@click.argument('conversation_id')
@click.pass_obj
def export_conversation(cvm: ConversationModel, user_id, persona_id, conversation_id, workdir_folder, filename):
    """Export a conversation as a jsonl file"""
    if filename is None:
        filename = f"{user_id}_{persona_id}_{conversation_id}.jsonl"

    output_file = os.path.join(workdir_folder if workdir_folder is not None else '.', filename)

    history = cvm.get_conversation_history(user_id, persona_id, conversation_id)
    history = [r.to_dict() for _, r in history.iterrows()]
    write_jsonl(history, output_file)

    click.echo(f"Conversation {conversation_id} for user {user_id} with {persona_id} has been exported to {output_file}. ({len(history)} messages)")

@cli.command()
@click.option('--workdir_folder', default=ENV_CONFIG['workdir_folder'], help='working directory')
@click.option('--filename', default=None, help='output file')
@click.pass_obj
def export_all(cvm: ConversationModel, workdir_folder, filename):
    """Export a conversation as a jsonl file"""
    if filename is None:
        filename = f"dump.jsonl"

    output_file = os.path.join(workdir_folder if workdir_folder is not None else '.', filename)

    history = cvm.collection.to_pandas()
    history.drop(columns=['index'], inplace=True)
    history = [r.to_dict() for _, r in history.iterrows()]
    write_jsonl(history, output_file)

    click.echo(f"All data has been exported to {output_file}. ({len(history)} messages)")

@cli.command()
@click.option('--user-id', default=ENV_CONFIG['user_id'], help='User ID for whom to apply the conversation')
@click.option('--persona-id', default=ENV_CONFIG['persona_id'], help='Persona ID for whom to apply the conversation')
@click.argument('conversation_filename')
@click.pass_obj
def import_conversation(cvm: ConversationModel, conversation_filename, user_id, persona_id):
    """Export a conversation as a jsonl file"""

    conversation_ids = defaultdict(int)
    data = read_jsonl(conversation_filename)
    for row in data:
        conversation_ids[row['conversation_id']] += 1
        if user_id is not None:
            row['user_id'] = user_id
        if persona_id is not None:
            row['persona_id'] = persona_id
        cvm.insert(**row)

    click.echo(f"Conversation {conversation_filename} has been imported.")
    
    for conversation_id, count in conversation_ids.items():
        click.echo(f"Conversation {conversation_id} has been imported. ({count} messages)")

@cli.command()
@click.argument('dump_filename')
@click.pass_obj
def import_dump(cvm: ConversationModel, dump_filename):
    """Import the contents of a conversation dump from a jsonl file"""

    conversation_ids = defaultdict(int)
    data = read_jsonl(dump_filename)
    for row in data:
        conversation_ids[row['conversation_id']] += 1
        cvm.insert(**row)

    click.echo(f"Conversation {dump_filename} has been imported.")
    
    for conversation_id, count in conversation_ids.items():
        click.echo(f"Conversation {conversation_id} has been imported. ({count} messages)")


@cli.command()
@click.option('--model-url', default=ENV_CONFIG['model_url'], help='URL for the OpenAI-compatible API')
@click.option('--api-key', default=ENV_CONFIG['api_key'], help='API key for the LLM service')
@click.option('--user-id', default=ENV_CONFIG['user_id'], help='User ID for the conversation')
@click.option('--persona-id', default=ENV_CONFIG['persona_id'], help='Persona ID for the conversation')
@click.option('--conversation-id', default=ENV_CONFIG['conversation_id'], help='Conversation ID (optional)')
@click.option('--max-tokens', default=ENV_CONFIG['max_tokens'], help='Maximum number of tokens for LLM response')
@click.option('--mood', default=ENV_CONFIG['persona_mood'], help='Mood for the chat')
@click.option('--temperature', default=ENV_CONFIG['temperature'], help='Temperature for LLM response')
@click.option('--test-mode', is_flag=True, help='Test mode')
@click.option('--top-n', default=ENV_CONFIG['top_n'], help='Top N for LLM response')
@click.pass_obj
def chat(cvm: ConversationModel, model_url, api_key, user_id, persona_id, conversation_id, max_tokens, temperature, mood, test_mode, top_n):
    """Start a new chat session"""
    localmodel = OpenAIProvider.from_url(model_url, api_key)

    config = ChatConfig.from_env()
    config.user_id = user_id
    config.persona_id = persona_id
    config.conversation_id = conversation_id or cvm.next_conversation_id(user_id=user_id, persona_id=persona_id)
    config.max_tokens = max_tokens
    config.temperature = temperature
    config.persona_mood = mood
    config.top_n = top_n

    # So the AI doesn't try and speak in the user's voice
    config.stop_sequences.append(f"{config.user_id}:")

    persona_file = os.path.join(config.persona_path, f"{persona_id}.json")
    if not os.path.exists(persona_file):
        click.echo(f"Persona {persona_id} not found in {config.persona_path}")
        return

    persona = Persona.from_json_file(persona_file)

    cm = ChatManager(llm=localmodel, cvm=cvm, config=config, persona=persona, clear_output=lambda: click.clear())
    save = not test_mode
    cm.chat_loop(save=save)

@cli.command()
@click.argument('user_id')
@click.argument('persona_id')
@click.option('--no-retry', is_flag=True, help='Do not prompt the user for input')
@click.pass_obj
def next_analysis(cvm: ConversationModel, user_id, persona_id, no_retry):
    """Run the analysis pipeline"""
    from .pipeline.analyze import analysis_pipeline
    from .pipeline.base import BasePipeline
    config = ChatConfig.from_env()

    try:
        next_conversation = cvm.next_analysis
        if next_conversation is None:
            click.echo("No conversations found for analysis")
            sys.exit(1)
        config.user_id = user_id
        config.persona_id = persona_id
        config.conversation_id = next_conversation
        config.no_retry = no_retry
        print(f"Next conversation ID: {config.conversation_id}")

        pipeline = BasePipeline.from_config(config)
        analysis_pipeline(self=pipeline)
    except Exception as e:
        print("Error running analysis pipeline")
        print(e)
        sys.exit(1)

@cli.command()
@click.argument('user_id')
@click.argument('persona_id')
@click.argument('conversation_id')
@click.argument('guidance', nargs=-1)
@click.option('--mood', default=None, help='The mood of the persona')
@click.option('--top-n', default=3, help='The top n for the general memory query')
@click.option('--no-retry', is_flag=True, help='Do not prompt the user for input')
@click.pass_context
def pipeline_analysis(ctx, user_id, persona_id, conversation_id, mood, guidance, top_n, no_retry):
    """Run the analysis pipeline"""
    from .pipeline.factory import pipeline_factory, BasePipeline, PIPELINE_ANALYSIS
    config = ChatConfig.from_env()
    config.user_id = user_id
    config.persona_id = persona_id
    config.conversation_id = conversation_id
    config.top_n = top_n
    config.no_retry = no_retry
    config.guidance = ' '.join(guidance)
    if mood:
        config.persona_mood = mood

    pipeline = BasePipeline.from_config(config)
    click.echo(f"Running analysis pipeline for {user_id} with {persona_id} in {conversation_id}")

    pipeline_factory(pipeline_type=PIPELINE_ANALYSIS)(self=pipeline)
    click.echo("Analysis pipeline complete")

    return

@cli.command()
@click.argument('persona_id')
@click.argument('conversation_id')
@click.option('--mood', default=None, help='The mood of the persona')
@click.option('--no-retry', is_flag=True, help='Do not prompt the user for input')
@click.option('--guidance', is_flag=True, help='Prompt for guidance for the conversation')
@click.argument('query', nargs=-1)
@click.pass_context
def pipeline_journal(ctx, persona_id, conversation_id, mood, query, no_retry, guidance):
    """Run the journal pipeline"""
    from .pipeline.journal import journal_pipeline, BasePipeline
    config = ChatConfig.from_env()
    config.persona_id = persona_id
    config.user_id = persona_id
    config.conversation_id = conversation_id
    config.no_retry = no_retry
    if guidance:
        value = click.prompt('Enter your guidance', type=str)
        config.guidance = value
        print(f"Guidance: {config.guidance}")
    if mood:
        config.persona_mood = mood

    pipeline = BasePipeline.from_config(config)
    query_text = ' '.join(query)

    journal_pipeline(self=pipeline, query_text=query_text)

@cli.command()
@click.argument('persona_id')
@click.argument('conversation_id')
@click.option('--mood', default=None, help='The mood of the persona')
@click.option('--no-retry', is_flag=True, help='Do not prompt the user for input')
@click.option('--guidance', is_flag=True, help='Prompt for guidance for the conversation')
@click.argument('query', nargs=-1)
@click.pass_context
def pipeline_chore(ctx, persona_id, conversation_id, mood, query, no_retry, guidance):
    """Run the chore pipeline"""
    from .pipeline.chore import chore_pipeline, BasePipeline
    config = ChatConfig.from_env()
    config.persona_id = persona_id
    config.user_id = persona_id
    config.conversation_id = conversation_id
    config.no_retry = no_retry
    if guidance:
        value = click.prompt('Enter your guidance', type=str)
        config.guidance = value
        print(f"Guidance: {config.guidance}")
    if mood:
        config.persona_mood = mood

    pipeline = BasePipeline.from_config(config)
    query_text = ' '.join(query)

    chore_pipeline(self=pipeline, query_text=query_text)

@cli.command()
@click.argument('persona_id')
@click.argument('conversation_id')
@click.option('--mood', default=None, help='The mood of the persona')
@click.option('--no-retry', is_flag=True, help='Do not prompt the user for input')
@click.option('--guidance', is_flag=True, help='Prompt for guidance for the conversation')
@click.option('--document', default=None, help='The document to use for the report')
@click.argument('query', nargs=-1)
@click.pass_context
def pipeline_report(ctx, persona_id, conversation_id, mood, query, no_retry, guidance, document):
    """Run the report pipeline"""
    from .pipeline.report import report_pipeline, BasePipeline
    config = ChatConfig.from_env()
    config.persona_id = persona_id
    config.user_id = persona_id
    config.conversation_id = conversation_id
    config.no_retry = no_retry
    if guidance:
        value = click.prompt('Enter your guidance', type=str)
        config.guidance = value
        print(f"Guidance: {config.guidance}")
    if mood:
        config.persona_mood = mood

    pipeline = BasePipeline.from_config(config)
    query_text = ' '.join(query)

    report_pipeline(self=pipeline, query_text=query_text, document_name=document)

if __name__ == '__main__':
    cli()
