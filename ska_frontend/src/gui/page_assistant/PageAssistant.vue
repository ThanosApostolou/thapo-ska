<script setup lang="ts">
import { onMounted, ref } from 'vue';
import CompChat from './CompChat.vue';
import ChatDialog from './ChatDialog.vue';
import { ChatPacketType, DtoChatPacket } from './dtos/dto_chat_packet';
import { ServiceAssistant } from './service_assistant';
import { DtoAssistantOptions } from './dtos/dto_fetch_assistant_options';
import type { DtoChatDetails } from './dtos/dto_chat_details';
import CompConfirmationDialog from '@/utils/ui/CompConfirmationDialog.vue';
import { DtoFetchChatMessagesRequest } from './dtos/dto_fetch_chat_messages';


// hooks
const chatPackets = ref<DtoChatPacket[]>([
  new DtoChatPacket({
    created_at: Date.now(),
    message_body: 'Please ask me anything related to this field',
    packet_type: ChatPacketType.ANSWER,
    context: []
  })
])

const isLoading = ref<boolean>(false);
const pageErrors = ref<string[]>([]);
const assistantOptions = ref<DtoAssistantOptions | null>(null);
const selectedUserChat = ref<DtoChatDetails | null>(null);
const userChatForAction = ref<DtoChatDetails | null>(null);
const chatDialog = ref<InstanceType<typeof ChatDialog> | null>(null)
const deleteDialogOpen = ref<boolean>(false);


// lifecycles
onMounted(async () => {
  await fetchAssistantOptions();

});

// functions
async function fetchAssistantOptions() {
  const lastChatId = selectedUserChat.value?.chat_id || 0;
  pageErrors.value = [];
  assistantOptions.value = null;
  selectedUserChat.value = null;
  isLoading.value = true;

  const result = await ServiceAssistant.fetchAssistantOptions();
  if (result.isErr()) {
    const errors = result.error;
    pageErrors.value = errors.packets.map(packet => packet.message);
    isLoading.value = false;
  } else {
    assistantOptions.value = result.data;
    const lastFoundChat = assistantOptions.value.user_chats.find(chat => chat.chat_id === lastChatId) || null;
    selectedUserChat.value = lastFoundChat;
    isLoading.value = false;
  }

}

async function fetchChatMessages(chat_id: number) {
  const request = new DtoFetchChatMessagesRequest({
    chat_id
  });
  isLoading.value = true;
  const result = await ServiceAssistant.fetchChatMessages(request);
  if (result.isErr()) {
    const errors = result.error;
    pageErrors.value = errors.packets.map(packet => packet.message);
    isLoading.value = false;
  } else {
    const data = result.data;
    chatPackets.value = [new DtoChatPacket({
      created_at: Date.now(),
      message_body: 'Please ask me anything related to this field',
      packet_type: ChatPacketType.ANSWER,
      context: []
    })];
    chatPackets.value = chatPackets.value.concat(data.chat_packets);
    isLoading.value = false;
  }
}

function onAddClicked() {
  userChatForAction.value = null;
  chatDialog.value?.showModal();
}

function onUpdateClicked(userChat: DtoChatDetails) {
  // selectedUserChat.value = userChat;
  userChatForAction.value = userChat;
  chatDialog.value?.showModal();
}

function onDeleteClicked(userChat: DtoChatDetails) {
  userChatForAction.value = userChat;
  deleteDialogOpen.value = true;
}

async function onRadioChange(event: Event) {
  console.log('event', event);
  console.log('selectedUserChat', selectedUserChat.value);
  await fetchChatMessages(selectedUserChat.value?.chat_id || 0);
}

function onCreateUpdate(chat_id: number) {
  console.log('success chat_id', chat_id);
  chatDialog.value?.close();
  fetchAssistantOptions();
}

async function onDeleteDialogAction(isConfirm: boolean) {
  if (!isConfirm) {
    deleteDialogOpen.value = false;
  } else {
    if (userChatForAction.value?.chat_id != null) {
      await ServiceAssistant.deleteChat(userChatForAction.value.chat_id);
      await fetchAssistantOptions();
    }
    deleteDialogOpen.value = false;
  }
}

</script>

<template>
  <template v-if="isLoading">
    <span class="loading loading-spinner text-primary"></span>
  </template>
  <template v-else>
    <div v-if="pageErrors.length > 0" class="flex flex-row flex-auto min-h-0">
      <p>{{ pageErrors }}</p>
    </div>

    <div v-else-if="assistantOptions != null" class="ska-page-container flex flex-row flex-auto min-h-0">
      <div class="ska-page-column bg-base-300 w-[16rem] break-words shadow">
        <button class="btn" @click="onAddClicked">
          <img src="/assets/icons/plus.svg" width="24" />Add Chat
        </button>

        <ChatDialog ref="chatDialog" :assistantOptions="assistantOptions" :user-chat-update="userChatForAction"
          @success="onCreateUpdate" />

        <div v-for="userChat in assistantOptions.user_chats" :key="userChat.chat_id || 0" class="form-control">
          <label class="label cursor-pointer">
            <input type="radio" :name="userChat.chat_name" :value="userChat" class="radio checked:bg-red-500"
              v-model="selectedUserChat" :disabled="isLoading" @change="onRadioChange" />
            <span class="label-text"><b>{{ userChat.chat_name }}</b> ({{ userChat.llm_model }})</span>
            <details class="dropdown dropdown-end">
              <summary class="btn btn-ghost btn-square"> <img src="/assets/icons/ellipsis-vertical.svg" width="24" />
              </summary>
              <ul class="p-2 shadow menu dropdown-content z-[1] bg-base-100 rounded-box w-48">
                <li @click="onUpdateClicked(userChat)">
                  <span><img src="/assets/icons/pencil.svg" width="24">Edit</span>
                </li>
                <li @click="onDeleteClicked(userChat)">
                  <span><img src="/assets/icons/trash.svg" width="24">Delete</span>
                </li>
              </ul>
            </details>
          </label>
        </div>
      </div>

      <div v-if="selectedUserChat != null" class="ska-page-column-flex flex">
        <CompChat :chat-packets="chatPackets" :selectedUserChat="selectedUserChat" />
      </div>
      <div v-else class="center p-2 prose max-w-none">
        <h4>Please Add or Select a chat in order to ask the Assistant some questions!</h4>
      </div>
    </div>

  </template>


  <CompConfirmationDialog v-if="deleteDialogOpen" :open="deleteDialogOpen" title="Delete Chat"
    @action="onDeleteDialogAction($event)">
    <div class="prose max-w-none">
      <h4>Are you sure you want to delete chat {{ userChatForAction?.chat_name }}?</h4>
    </div>
  </CompConfirmationDialog>
</template>