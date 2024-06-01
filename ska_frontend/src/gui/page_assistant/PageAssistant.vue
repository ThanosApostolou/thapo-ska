<script setup lang="ts">
import { onMounted, ref } from 'vue';
import CompChat from './CompChat.vue';
import { ChatPacketType, DtoChatPacket } from './dtos/dto_chat_packet';
import { ServiceAssistant } from './service_assistant';
import { DtoAssistantOptions, DtoLlmData } from './dtos/dto_fetch_assistant_options';

// hooks
const chatPackets = ref<DtoChatPacket[]>([
  new DtoChatPacket({
    timestamp: Date.now(),
    value: 'Please ask me anything related to this field',
    packet_type: ChatPacketType.ANSWER
  })
])

const pageErrors = ref<string[]>([]);
const assistantOptions = ref<DtoAssistantOptions | null>(null);
const selectedLlm = ref<DtoLlmData | null>(null);
const prompt = ref<string>('');
const isEditPrompt = ref<boolean>(false);

// lifecycles
onMounted(async () => {
  await fetchAssistantOptions();

});

// functions
async function fetchAssistantOptions() {
  const result = await ServiceAssistant.fetchAssistantOptions();
  if (result.isErr()) {
    const errors = result.error;
    pageErrors.value = errors.packets.map(packet => packet.message);
  } else {
    assistantOptions.value = result.data;
    selectedLlm.value = assistantOptions.value.llms[0];
    prompt.value = selectedLlm.value.default_prompt;
  }

}

function onSelectChange() {
  isEditPrompt.value = false;
  prompt.value = selectedLlm.value?.default_prompt || '';

}

</script>

<template>
  <div v-if="pageErrors.length > 0" class="flex flex-row flex-auto min-h-0">
    <p>{{ pageErrors }}</p>
  </div>
  <div v-else-if="assistantOptions != null" class="flex flex-row flex-auto min-h-0">
    <div class="ska-page-column bg-base-300 max-w-64 break-words">
      <button class="btn">
        <img src="/assets/icons/plus.svg" width="24" />Add Chat
      </button>

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
    </div>

    <div v-if="selectedLlm != null" class="ska-page-column-flex flex">
      <CompChat :chat-packets="chatPackets" :selectedLlm="selectedLlm" :prompt="prompt" :isEditPrompt="isEditPrompt" />
    </div>
  </div>
</template>