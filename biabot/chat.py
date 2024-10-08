# assignment/chat.py

import time
from typing import List, Dict, Callable, Tuple, Optional, Generator

from .constants import ROLE_ASSISTANT, ROLE_USER
from .documents import Library
from .llm import LLMProvider, ChatConfig
from .models.conversation import ConversationModel
from .persona import Persona


HELP = """Commands:
- (b)ack: Go back to the previous message
- (h)elp: Show this help message
- new: Start a new chat
- (p)rompt <message>: Update the system message
- redraw: Redraw the screen
- (d)ocument <name>: Set the current document
- (l)ist: List the documents in the library
- (r)etry: Retry the previous user input
- (s)earch <query>: Search your documents
- top <n>: Set the top n results to use
- temp <n>: Set the temperature
- (q)uit/exit: End the chat
"""

def insert_at_fold(turns: List[Dict[str, str]], content: str, fold_depth: int = 4) -> List[Dict[str, str]]:
    """
    Inserts content at the first user turn after the specified fold depth in a list of chat turns.
    
    Args:
        turns (List[Dict[str, str]]): The list of chat turns.
        content (str): The content to insert.
        fold_depth (int): The fold depth to search for the first user turn. Defaults to 4.
    
    Returns:
        List[Dict[str, str]]: The updated list of chat turns with the content inserted.
    """
        
    # We work from the end, counting the number of user turns until we find the first one, or hit the fold depth
    depth = 0
    ix = 0
    while True:
        if ix >= len(turns) - 1:
            break
        if turns[ix]["role"] == "user":
            depth += 1
            if depth >= fold_depth:
                break
        ix += 1
        
    print(f"Inserting at fold depth {fold_depth} at index {ix}")
    # Copy our turns
    new_turns = [*turns]
    # Get our user turn
    target = new_turns[ix]
    # Insert our content
    new_turns[ix] = {"role": target['role'], "content": f"{content}{target['content']}"}

    return new_turns

class ChatTurnStrategy:
    """
    Generates a chat turn for the user input, augmenting the response with information from the database.
    
    This method is responsible for generating the chat turns that will be passed to the chat completion API. It retrieves relevant memories from the conversation model and inserts them into the chat history at the appropriate fold depth.
    
    Args:
        user_input (str): The current user input.
        history (List[Dict[str, str]]): The chat history up to this point.
    
    Returns:
        List[Dict[str, str]]: The updated chat turns, including the user input and any inserted memories.
    """
        
    def __init__(self, chat : 'ChatManager'):
        self.chat = chat

    def user_turn_for(self, user_input: str, history: List[Dict[str, str]] = []) -> Dict[str, str]:
        """
        Generate a user turn for a chat session.
        
        This is what will be stored in the history.
        
        Args:
            user_input (str): The user input.
            history (List[Dict[str, str]]): The chat history.

        Returns:
            Dict[str, str]: The user turn, in the format {"role": "user", "content": user_input}.
        """
        return {"role": "user", "content": user_input}

    def get_conscious_memory(self, query: Optional[str] = None, user_queries: List[str] = [], assistant_queries: List[str] = []) -> str:
        """
        Retrieves the conscious memory content to be included in the chat response.
        
        The conscious memory content includes the persona's thoughts, as well as relevant memories from the conversation history. It also includes any relevant documents that have been revealed to the user.
        
        Args:
            query (Optional[str]): The current user query, used to filter the retrieved memories.
            user_queries (List[str]): The history of user queries, used to retrieve relevant memories.
            assistant_queries (List[str]): The history of assistant queries, used to retrieve relevant memories.
        
        Returns:
            str: The conscious memory content, formatted as a string to be included in the chat response.
        """
                
        conscious_content = []
        conscious = self.chat.cvm.get_conscious(self.chat.config.persona_id, top_n=self.chat.config.recall_size)
        conscious_content.append(f"**Hello {self.chat.persona.name}, this is your conscious mind. I'm hijacking the stream to give you Memories to use**")
        for thought in self.chat.persona.thoughts:
            conscious_content.append(f"- {thought}")

        seen_docs = set()
        for _, row in conscious.iterrows():
            if row['doc_id'] in seen_docs:
                continue
            seen_docs.add(row['doc_id'])
            conscious_content.append(f"- {row['date']}: {row['content']}")
            print(f"CMemory: {row['conversation_id']}/{row['date']}/{row['doc_id']}")

        if query is not None:
            filter_text = f"conversation_id != '{self.chat.config.conversation_id}'"
            top_n = self.chat.config.memory_window - len(conscious)
            a_top = top_n // 2
            u_top = top_n - a_top

            a_results = self.chat.cvm.query(assistant_queries, filter_text=filter_text, filter_doc_ids=seen_docs, top_n=a_top, filter_metadocs=True)
            for _, row in a_results.reset_index().iterrows():
                conscious_content.append(f"- {row['date']}: {row['content']}")
                seen_docs.add(row['doc_id'])
                print(f"AMemory: {row['conversation_id']}/{row['date']}/{row['doc_id']}")

            u_results = self.chat.cvm.query(user_queries, filter_text=filter_text, filter_doc_ids=seen_docs, top_n=u_top, filter_metadocs=True)
            for _, row in u_results.reset_index().iterrows():
                conscious_content.append(f"- {row['date']}: {row['content']}")
                seen_docs.add(row['doc_id'])
                print(f"UMemory: {row['conversation_id']}/{row['date']}/{row['doc_id']}")

            conscious_content.append("**All of these memories were in the past, and not part of the current conversation**")


        conscious_content.append("**End of Memories**\n\n")
        if self.chat.current_doucment is not None:
            print(f"Current Document: {self.chat.current_doucment}")
            document_contents = self.chat.library.read_document(self.chat.current_doucment)
            conscious_content.append(f"==={self.chat.config.user_id} is revealing a Document to you===")
            conscious_content.append(f"Document Name: [{self.chat.current_doucment}]")
            conscious_content.append(f"Document Length: {len(document_contents)}")
            conscious_content.append("===Begin Document===")
            conscious_content.append(document_contents)
            conscious_content.append("===End of Document===")


        return "\n\n".join(conscious_content)
        
    def chat_turns_for(self, user_input: str, history: List[Dict[str, str]] = []) -> List[Dict[str, str]]:
        """
        Generate a chat session, augmenting the response with information from the database.

        This is what will be passed to the chat complletion API.

        Args:
            user_input (str): The user input.
            history (List[Dict[str, str]]): The chat history.
            
        Returns:
            List[Dict[str, str]]: The chat turns, in the alternating format [{"role": "user", "content": user_input}, {"role": "assistant", "content": assistant_turn}].
        """

        fold_consciousness = 4
        assistant_turn_history = [r['content'] for r in history if r['role'] == 'assistant'][::-1]
        user_turn_history = [r['content'] for r in history if r['role'] == 'user'][::-1]
        user_turn_history.append(user_input)
        consciousness = self.get_conscious_memory(
                query=user_input,
                user_queries=user_turn_history,
                assistant_queries=assistant_turn_history,
                )

        turn = [*history, {"role": "user", "content": user_input + "\n\n"}]
        
        if len(consciousness) > 0:
            turn = insert_at_fold(turn, consciousness, fold_consciousness)

        return turn


class ChatManager:
    def __init__(self, llm: LLMProvider, cvm: ConversationModel, config: ChatConfig, persona: Persona, clear_output=Callable[[], None]):
        self.llm = llm
        self.cvm = cvm
        self.config = config
        self.clear_output = clear_output
        self.persona = persona
        self.library = Library(documents_dir=config.documents_dir)
        self.current_doucment : Optional[str] = None
        self.chat_strategy = ChatTurnStrategy(self)

        self.running = False
        self.history : List[Dict[str, str]] = []

    def render_conversation(self, messages: List[Dict[str, str]]) -> None:
        if self.clear_output is not None:
            self.clear_output()
        for message in messages:
            rolename = message['role'].capitalize()
            print(f"{rolename}: {message['content']}\n")
        print(flush=True)

    def add_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        self.history.clear()

    def render_conversation(self, messages: List[Dict[str, str]]) -> None:
        self.clear_output()
        for message in messages:
            print(f"{message['role']}: {message['content']}\n")
        print(flush=True)

    def new_conversation(self) -> None:
        self.config.conversation_id = self.cvm.next_conversation_id(self.config.user_id)

    def get_system_prompt(self):
        return self.persona.system_prompt(mood=self.config.persona_mood, location=f"You are at {self.config.persona_location}", user_id=self.config.user_id)

    def handle_user_input(self) -> Generator[Tuple[str, str, Optional[str]], None, None]:
        # Next, we get the user input, and handle special commands
        user_input = input("You (h for help): ").strip()

        # If the user input is empty, we skip this iteration
        if not user_input or user_input.strip() == '':
            print("Type 'h' for help, 'q' to quit.")
            yield 'pass', 'No input provided', None
            return

        # Handle single word commands
        lowered = user_input.lower()
        if lowered in ['q', 'quit', 'exit']:
            yield 'quit', None, None
            return

        if lowered in ['b', 'back']:
            if len(self.history) >= 2:
                self.history = self.history[:-2]
            yield 'back', 'Back one turn', None
            return

        if lowered == 'new':
            self.clear_history()
            yield 'new', 'New chat started', None
            return

        if lowered == 'redraw':
            yield 'redraw', 'Redrew the screen', None
            return

        if lowered in ['h', 'help']:
            yield 'help', 'Help', None
            return

        if lowered in ['l', 'list']:
            yield 'list', 'List of documents', None
            return

        # Handle multi-word commands
        multi = lowered.split()
        # 'prompt' allows the user to update the system message
        if len(multi) > 1 and multi[0] in ['p', 'prompt']:
            # TODO re-add persona.
            self.config.system_message = ' '.join(multi[1:])
            yield 'prompt', 'System Prompt Updated', None
            return

        # Set our top-n
        if len(multi) > 1 and multi[0] in ['top']:
            try:
                intval = int(multi[1])
                if intval > 0:
                    self.config.top_n = intval
                    yield 'top_n_set', f'Top N set to {self.config.top_n}', None
                else:
                    yield 'error', 'Invalid top N value', None
            except ValueError:
                yield 'error', 'Invalid top N value', None

            return

        # Set our temperature
        if len(multi) > 1 and multi[0] in ['temp']:
            try:
                self.config.temperature = float(multi[1])
                yield 'temperature_set', f'Temperature set to {self.config.temperature}', None
            except ValueError:
                yield 'invalid_temperature', 'Invalid temperature value', None

            return

        # Search for documents
        if len(multi) > 1 and multi[0] in ['s', 'search']:
            documents = self.cvm.query(' '.join(multi[1:]))
            if documents.empty:
                yield 'no_results', 'No documents found', None
            else:
                for _, row in documents.reset_index()[::-1].iterrows():
                    cid = row['conversation_id']
                    dist = row['score']
                    c = row['content'][:80]
                    print(f"Document {cid} (distance: {dist:.2f})")
                    print(c)
                yield 'found_document', None, None
            return
        
        if len(multi) > 0 and multi[0] in ['d', 'document']:
            try:
                if len(multi) > 1:
                    document = multi[1]
                    if self.library.exists(document):
                        self.current_doucment = document
                        yield 'document_set', f'Document set to {self.current_doucment}', None
                    else:
                        yield 'document_set', f'Document does not exist, current document {self.current_doucment}.', None
                else:
                    self.current_doucment = None
                    yield 'document_set', 'Document cleared', None
            except Exception as e:
                yield 'error', f'Error setting document: {e}', None
            return

        if lowered in ['r', 'retry']:
            if len(self.history) >= 2:
                user_input = self.history[-2]["content"]
                self.history = self.history[:-2]
                yield 'retry', 'Retried the last turn', None
            else:
                yield 'pass', 'No previous turn to retry', None

        yield 'user', user_input, int(time.time())

    def run_once(self) -> Generator[Tuple[str, str, Optional[int]], None, None]:
        # Render our current turn - we could optionally just include only the history, but for debugging purposes,
        # we'll render the entire turn
        self.render_conversation(self.history)

        user_input = None
        for event in self.handle_user_input():
            action, message, _ = event
            
            yield event

            if action == 'user':
                user_input = message

        if user_input is None:
            return

        user_turn = self.chat_strategy.user_turn_for(user_input)
        chat_turns = self.chat_strategy.chat_turns_for(user_input, self.history)

        if self.config.debug:
            self.render_conversation(chat_turns)

        print(f"Assistant: ", end='', flush=True)

        chunks = []

        for t in self.llm.stream_turns(chat_turns, self.config):
            if t is not None:
                print(t, end='', flush=True)
                chunks.append(t)
            else:
                print('', flush=True)

        response = ''.join(chunks)

        yield 'assistant', response, int(time.time())

        ui = input("** -=[ <enter> or (r)etry ]=- **")
        if ui == 'r':
            yield 'pass', 'Retrying', None
            return

        self.add_history(**user_turn)
        self.add_history("assistant", response)

        yield 'continue', None, None
        return

    def chat_loop(self, save: bool=True) -> None:
        history = self.cvm.get_conversation_history(conversation_id=self.config.conversation_id).sort_values(['date', 'sequence_no', 'branch']).reset_index(drop=True)
        if history.empty:
            self.sequence_no = 0
            self.branch = 0
            self.history = []
        else:
            last = history.iloc[-1]
            self.sequence_no = last['sequence_no'] + 1
            self.branch = last['branch']
            self.history = history[['role', 'content']].to_dict(orient='records')

        self.config.system_message = self.get_system_prompt()

        self.running = True
        while self.running:
            try:
                enter = True
                user_input : Optional[str] = None
                usertime : Optional[int] = None

                assistant_response : Optional[str] = None
                assttime : Optional[int] = None

                for event in self.run_once():
                    result, message, ts = event
                    
                    if result == 'quit':
                        self.running = False
                        enter = False
                    elif result == 'redraw':
                        enter = False
                    elif result == 'new':
                        self.branch = 0
                        self.sequence_no = 0
                        self.history = []
                        self.config.system_message = self.get_system_prompt()
                        self.config.conversation_id = self.cvm.next_conversation_id(self.config.user_id, self.config.persona_id)
                        enter = False
                    elif result == 'help':
                        print()
                        print(HELP)
                    elif result == 'back' or result == 'retry':
                        self.branch += 1
                    elif result == 'user':
                        user_input = message
                        usertime = ts
                        print('user', ts, len(message))
                    elif result == 'assistant':
                        assistant_response = message
                        assttime = ts
                        print('assistant', ts, len(message))
                    elif result == 'list':
                        print(f"Library {self.library.documents_dir}:")
                        for f, t, s in self.library.list_documents:
                            print('  ', f, t, s)
                    elif result == 'continue':
                        enter = False

                        if save:
                            ut = user_input
                            ar = assistant_response
                            self.cvm.insert(conversation_id=self.config.conversation_id, user_id=self.config.user_id, persona_id=self.config.persona_id,
                                            branch=self.branch, sequence_no=self.sequence_no, role=ROLE_USER, content=ut, timestamp=usertime,
                                            inference_model=self.llm.model)
                            self.cvm.insert(conversation_id=self.config.conversation_id, user_id=self.config.user_id, persona_id=self.config.persona_id,
                                            branch=self.branch, sequence_no=self.sequence_no + 1, role=ROLE_ASSISTANT, content=ar, timestamp=assttime,
                                            inference_model=self.llm.model)
                            self.sequence_no += 2
                    else:
                        if message is not None:
                            print(f"{result}: {message}")
                        else:
                            print(f"{result}", end='')

                if enter:
                    print()
                    input("-=[ Hit <enter> to continue... ]=-")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"An error occurred: {e}")

                print()
                input("-=[ Hit <enter> to continue... ]=-")
            
        self.running = False
        print("Chat session ended.")

    def __repr__(self):
        return f"ChatManager(history={len(self.history)} documents={self.cvm.collection.count_rows()} config={self.config})"
