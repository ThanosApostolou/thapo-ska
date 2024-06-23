<!-- eslint-disable vue/no-mutating-props -->
<script setup lang="ts">
import { ref } from 'vue';
import { ChatPacketType, DtoChatPacket } from './dtos/dto_chat_packet';
import { AskAssistantQuestionRequest } from './dtos/dto_ask_assistant_question';
import { ServiceAssistant } from './service_assistant';
import type { DtoChatDetails } from './dtos/dto_chat_details';

// hooks
const props = defineProps<{
  chatPackets: DtoChatPacket[],
  selectedUserChat: DtoChatDetails,
}>();
const question = ref<string>('');

async function onSubmit() {
  const request = new AskAssistantQuestionRequest({
    chat_id: props.selectedUserChat.chat_id || -1,
    question: question.value,
  });

  const response = await ServiceAssistant.askAssistantQuestion(request);
  if (response.isOk()) {
    const data = response.data;
    props.chatPackets.push(data.question);

    props.chatPackets.push(data.answer);
  }

}
</script>

<template>
  <div class="flex flex-col flex-auto items-stretch">
    <div class="ska-page-column-flex bg-error">
      <template v-for="(chatPacket, index) of props.chatPackets" :key="index">
        <div v-if="chatPacket.packet_type === ChatPacketType.ANSWER" class="chat chat-start">
          <div class="chat-bubble">{{ chatPacket.message_body }}</div>
        </div>
        <div v-if="chatPacket.packet_type === ChatPacketType.QUESTION" class="chat chat-end">
          <div class="chat-bubble">{{ chatPacket.message_body }}</div>
        </div>
      </template>
    </div>

    <div class="ska-page-column">
      <form class="form-control" @submit.prevent="onSubmit">
        <label class="label w-full">
          <input type="text" placeholder="Ask your question" class="input input-bordered input-primary w-full mr-1"
            v-model="question" />
          <button type="submit" class="btn btn-outline btn-primary">
            <img src="/assets/icons/paper-airplane.svg" height="32" width="32" />
          </button>
        </label>
      </form>
    </div>
  </div>
</template>