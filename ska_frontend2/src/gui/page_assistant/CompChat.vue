<!-- eslint-disable vue/no-mutating-props -->
<script setup lang="ts">
import { ref } from 'vue';
import { ChatPacketType, DtoChatPacket } from './dtos/dto_chat_packet';
import { AskAssistantQuestionRequest } from './dtos/dto_ask_assistant_question';
import { ServiceAssistant } from './service_assistant';

// hooks
const props = defineProps<{
  chatPackets: DtoChatPacket[]
}>();
// const chat_packets = ref<DtoChatPacket[]>([])
const question = ref<string>('');

async function onSubmit() {
  const request = new AskAssistantQuestionRequest({
    question: question.value,
    llm_model: "gpt2",
    prompt_template: null,
  });

  const newTimestamp = Date.now();
  const answer = await ServiceAssistant.askAssistantQuestion(request);
  if (answer.isOk()) {
    const data = answer.data;
    props.chatPackets.push(new DtoChatPacket({
      timestamp: newTimestamp,
      value: question.value,
      packet_type: ChatPacketType.QUESTION,
    }));

    props.chatPackets.push(new DtoChatPacket({
      timestamp: Date.now(),
      value: data.answer,
      packet_type: ChatPacketType.ANSWER,
    }));
  }

}
</script>

<template>
  <div class="flex flex-col flex-auto items-stretch">
    <div class="ska-page-column-flex bg-error">
      <template v-for="(chatPacket, index) of props.chatPackets" :key="index">
        <div v-if="chatPacket.packet_type === ChatPacketType.ANSWER" class="chat chat-start">
          <div class="chat-bubble">{{ chatPacket.value }}</div>
        </div>
        <div v-if="chatPacket.packet_type === ChatPacketType.QUESTION" class="chat chat-end">
          <div class="chat-bubble">{{ chatPacket.value }}</div>
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