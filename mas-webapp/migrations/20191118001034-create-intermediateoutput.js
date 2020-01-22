'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('IntermediateOutput', {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      RunId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'RunDetail',
          key: 'Id'
        }
      },
      RegsDataframeJSON: Sequelize.TEXT,
      BaseDataframeJSON: Sequelize.TEXT,
      DependentJSON: Sequelize.TEXT
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('IntermediateOutput');
  }
};
