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
const selectedChatPacket = ref<DtoChatPacket | null>(null);
const contextDialog = ref<HTMLDialogElement | null>(null);
const isWaitingAsync = ref<boolean>(false);

async function onSubmit() {
  const request = new AskAssistantQuestionRequest({
    chat_id: props.selectedUserChat.chat_id || -1,
    question: question.value,
  });

  isWaitingAsync.value = true;
  const response = await ServiceAssistant.askAssistantQuestion(request);
  if (response.isOk()) {
    const data = response.data;
    props.chatPackets.push(data.question);

    props.chatPackets.push(data.answer);
    isWaitingAsync.value = false;
  } else {
    isWaitingAsync.value = false;
  }
}

function onShowContextClick(chatPacket: DtoChatPacket) {
  selectedChatPacket.value = chatPacket;
  contextDialog.value?.showModal();
}
</script>

<template>
  <div class="flex flex-col flex-auto items-stretch">
    <div class="ska-page-column-flex bg-error">
      <template v-for="(chatPacket, index) of props.chatPackets" :key="index">
        <div v-if="chatPacket.packet_type === ChatPacketType.ANSWER" class="chat chat-start">
          <div class="chat-bubble">
            {{ chatPacket.message_body }}
            <div class="tooltip" data-tip="show context">
              <button :disabled="isWaitingAsync" v-if="chatPacket.context.length > 0" class="indicator-item badge"
                @click="onShowContextClick(chatPacket)"><img src="/assets/icons/bors-3.svg" width="16" /></button>
            </div>
          </div>
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
          <button type="submit" :disabled="isWaitingAsync" class="btn btn-outline btn-primary">
            <span v-if="isWaitingAsync" class="loading loading-spinner loading-sm"></span>
            <img src="/assets/icons/paper-airplane.svg" height="32" width="32" />
          </button>
        </label>
      </form>
    </div>
  </div>

  <dialog v-if="selectedChatPacket != null" ref="contextDialog" id="contextDialog" class="modal">
    <div class="modal-box w-11/12 max-w-6xl h-5/6">
      <h3 class="text-2xl font-bold">Context</h3>
      <br>
      <template v-for="(document, index) of selectedChatPacket.context" :key="index">
        <p class="text-xl font-bold">Context Document {{ index }}:</p>
        <p class="text-lg font-bold">Content:</p>
        <p>{{ document.page_content }}</p>
        <p class="text-lg font-bold">Source:</p>
        <p>{{ document.metadata["source"] }}</p>
        <hr>
        <br>
      </template>
      <p class="py-4">Press ESC key or click outside to close</p>
    </div>
    <form method="dialog" class="modal-backdrop">
      <button>close</button>
    </form>
  </dialog>
</template>