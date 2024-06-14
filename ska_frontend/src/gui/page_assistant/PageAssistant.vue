<script setup lang="ts">
import { onMounted, ref } from 'vue';
import CompChat from './CompChat.vue';
import ChatDialog from './ChatDialog.vue';
import { ChatPacketType, DtoChatPacket } from './dtos/dto_chat_packet';
import { ServiceAssistant } from './service_assistant';
import { DtoAssistantOptions, DtoLlmData } from './dtos/dto_fetch_assistant_options';
import type { DtoChatDetails } from './dtos/dto_chat_details';

// hooks
const chatPackets = ref<DtoChatPacket[]>([
  new DtoChatPacket({
    timestamp: Date.now(),
    value: 'Please ask me anything related to this field',
    packet_type: ChatPacketType.ANSWER
  })
])

const isLoading = ref<boolean>(false);
const pageErrors = ref<string[]>([]);
const assistantOptions = ref<DtoAssistantOptions | null>(null);
const selectedLlm = ref<DtoLlmData | null>(null);
const selectedUserChat = ref<DtoChatDetails | null>(null);
const prompt = ref<string>('');
const isEditPrompt = ref<boolean>(false);
const chatDialog = ref<InstanceType<typeof ChatDialog> | null>(null)


// lifecycles
onMounted(async () => {
  await fetchAssistantOptions();

});

// functions
async function fetchAssistantOptions() {
  pageErrors.value = [];
  assistantOptions.value = null;
  selectedLlm.value = null;
  selectedUserChat.value = null;
  prompt.value = '';
  isLoading.value = true;
  const result = await ServiceAssistant.fetchAssistantOptions();
  if (result.isErr()) {
    const errors = result.error;
    pageErrors.value = errors.packets.map(packet => packet.message);
    isLoading.value = false;
  } else {
    assistantOptions.value = result.data;
    selectedLlm.value = assistantOptions.value.llms[0];
    selectedUserChat.value = null;
    prompt.value = selectedLlm.value.default_prompt;
    isLoading.value = false;
  }

}

function onSelectChange() {
  isEditPrompt.value = false;
  prompt.value = selectedLlm.value?.default_prompt || '';

}

function onCreateUpdate(chat_id: number) {
  console.log('success chat_id', chat_id);
  chatDialog.value?.close();
  fetchAssistantOptions();
}

</script>

<template>
  <div v-if="isLoading">
    <span class="loading loading-spinner text-primary"></span>
  </div>
  <div v-else>
    <div v-if="pageErrors.length > 0" class="flex flex-row flex-auto min-h-0">
      <p>{{ pageErrors }}</p>
    </div>
    <div v-else-if="assistantOptions != null" class="flex flex-row flex-auto min-h-0">
      <div class="ska-page-column bg-base-300 max-w-64 break-words">
        <button class="btn" @click="chatDialog?.showModal()">
          <img src="/assets/icons/plus.svg" width="24" />Add Chat
        </button>

        <ChatDialog ref="chatDialog" :assistantOptions="assistantOptions" :chat-details="null"
          @success="onCreateUpdate" />



        <label for="llms" class="form-control">Choose a LLM:</label>
        <select v-model="selectedLlm" name="llms" id="llms" class="select" @change="onSelectChange()">
          <option v-for="llm of assistantOptions.llms" :key="llm.name" :value="llm">
            {{ llm.name }}
          </option>
        </select>

        <div class="form-control">
          <label class="label cursor-pointer">
            <span class="label-text">Edit prompt</span>
            <input type="checkbox" v-model="isEditPrompt" class="checkbox" />
          </label>
        </div>

        <textarea v-model="prompt" placeholder="Prompt" :disabled="!isEditPrompt"
          class="textarea textarea-bordered textarea-xs w-full max-w-xs" rows="16">
      </textarea>

        <div v-for="userChat in assistantOptions.user_chats" :key="userChat.chat_id || 0" class="form-control">
          <label class="label cursor-pointer">
            <input type="radio" :name="userChat.chat_name" :value="userChat" class="radio checked:bg-red-500"
              v-model="selectedUserChat" :disabled="isLoading" />
            <span class="label-text">{{ userChat.chat_name }}</span>
            <details class="dropdown dropdown-end">
              <summary class="btn btn-ghost btn-square"> <img src="/assets/icons/ellipsis-vertical.svg" width="24" />
              </summary>
              <ul class="p-2 shadow menu dropdown-content z-[1] bg-base-100 rounded-box">
                <li><a>Item 1</a></li>
                <li><a>Item 2</a></li>
              </ul>
            </details>
          </label>
        </div>
      </div>

      <div v-if="selectedLlm != null" class="ska-page-column-flex flex">
        <CompChat :chat-packets="chatPackets" :selectedLlm="selectedLlm" :prompt="prompt"
          :isEditPrompt="isEditPrompt" />
      </div>
    </div>
  </div>
</template>